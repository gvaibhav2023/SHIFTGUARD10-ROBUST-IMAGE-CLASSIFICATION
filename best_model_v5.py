
# =============================================================================
# ShiftGuard10 — best_model_v5.py
# Strategy: Soft Label Knowledge Distillation
#
# Moving past 0.90 requires breaking "Confirmation Bias". Instead of using
# hard labels (e.g. "100% Frog"), the Teacher model outputs its full
# uncertainty profile (e.g. "85% Frog, 10% Toad, 5% Truck"). The student
# trains using KL-Divergence against this full continuous distribution.
# 
# What's new:
#   ✦ Soft Pseudo-Labels (.pt dict): Saved alongside the .csv and used natively
#     during training to teach the network how to handle ambiguity.
#   ✦ Temperature Scaling: Flattens the teacher's confident predictions slightly
#     to preserve the entropy of the probability distribution.
#   ✦ Retains massive 50-pass Test-Time AugMix.
# =============================================================================

import os, io, random, time, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
warnings.filterwarnings('ignore')

# ── 0. GPU Setup ──────────────────────────────────────────────────────────────
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS  = torch.cuda.device_count()
USE_MULTI = NUM_GPUS > 1
print(f"GPUs: {NUM_GPUS}")
for i in range(NUM_GPUS):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

# ── 1. Config ─────────────────────────────────────────────────────────────────
class CFG:
    seed         = 42
    num_classes  = 10
    batch_size   = 128
    num_workers  = 4
    epochs       = 100          # fine-tuning from v4 checkpoint
    warmup_eps   = 2
    swa_start    = 60
    swa_lr       = 0.001

    lr           = 0.005        # low LR because teacher is highly tuned
    momentum     = 0.9
    weight_decay = 5e-4
    nesterov     = True
    grad_clip    = 5.0

    T0           = 25
    T_mult       = 1
    eta_min      = 1e-6
    val_frac     = 0.05

    randaug_n    = 2
    randaug_m    = 12
    mixup_alpha  = 0.4
    cutmix_alpha = 1.0
    mix_prob     = 0.5
    label_smooth = 0.1
    lambda_jsd   = 12.0

    # Pseudo-labeling
    pseudo_epoch    = 0         
    pseudo_conf     = 0.85      
    pseudo_tta      = 15        
    pseudo_temp     = 1.5       # NEW: Temperature to flatten soft-labels

    tta_n        = 50           # Massively diverse 50-pass TTA

    # IMPORTANT: Update this path to your v4 checkpoint (the one that scored 0.904 LB)
    pretrained_ckpt = Path('/kaggle/input/models/kartikeyguptaiitk/best-v4/pytorch/default/1/best_model_v4.pth')

    if os.path.exists('/kaggle/input'):
        base_dir = Path('/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge')
        out_dir  = Path('/kaggle/working')
    else:
        base_dir = Path(r'c:\Users\karti\Downloads\shiftgaurd\shift-guard-10-robust-image-classification-challenge')
        out_dir  = Path(r'c:\Users\karti\Downloads\shiftgaurd')

    train_dir  = base_dir / 'train_images'
    test_dir   = base_dir / 'test_images'
    train_csv  = base_dir / 'train_labels.csv'
    sample_sub = base_dir / 'sample_submission.csv'

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
MEAN    = [0.4914, 0.4822, 0.4465]
STD     = [0.2470, 0.2435, 0.2616]

print(f"Batch    : {CFG.batch_size} (combined→{CFG.batch_size*2}/step)")
print(f"Epochs   : {CFG.epochs} | SWA from {CFG.swa_start}")
print(f"Pseudo   : Soft Labels, temp={CFG.pseudo_temp}, {CFG.pseudo_tta} passes")
print(f"TTA      : {CFG.tta_n} passes")
est = CFG.epochs * 130 / 60 if USE_MULTI else CFG.epochs * 270 / 60
print(f"Est.time : {est:.0f} min ({est/60:.1f} hrs)\n")

# ── 2. Reproducibility ────────────────────────────────────────────────────────
def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(CFG.seed)

# ── 3. Corruption augmentations ───────────────────────────────────────────────
def gaussian_noise(img, std_range=(0.02, 0.12)):
    arr = np.array(img, dtype=np.float32) / 255.
    std = random.uniform(*std_range)
    arr = np.clip(arr + np.random.randn(*arr.shape) * std, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def motion_blur(img, max_size=5):
    size = random.choice([3, 5])
    return img.filter(ImageFilter.BoxBlur(size // 2))

def pixelate(img, factor=None):
    w, h = img.size
    f = factor or random.randint(2, 4)
    small = img.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)

def jpeg_compress(img, quality_range=(20, 60)):
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

CORRUPTION_OPS = [gaussian_noise, motion_blur, pixelate, jpeg_compress]

def apply_corruption(img):
    if random.random() < 0.3:
        return random.choice(CORRUPTION_OPS)(img)
    return img

# ── 4. AugMix ─────────────────────────────────────────────────────────────────
def _int(level, maxval): return int(level * maxval / 10)
def _flt(level, maxval): return float(level) * maxval / 10.

AUG_OPS = [
    lambda img, l: ImageOps.autocontrast(img),
    lambda img, l: ImageOps.equalize(img),
    lambda img, l: ImageOps.posterize(img, max(1, 4 - _int(l, 4))),
    lambda img, l: ImageOps.solarize(img, 256 - _int(l, 256)),
    lambda img, l: img.rotate(_int(l, 30) * random.choice([-1, 1]), fillcolor=(128,128,128)),
    lambda img, l: img.transform(img.size, Image.AFFINE, (1, _flt(l,.3)*random.choice([-1,1]), 0, 0, 1, 0), fillcolor=(128,128,128)),
    lambda img, l: img.transform(img.size, Image.AFFINE, (1, 0, 0, _flt(l,.3)*random.choice([-1,1]), 1, 0), fillcolor=(128,128,128)),
    lambda img, l: ImageEnhance.Color(img).enhance(_flt(l, 1.8) + 0.1),
    lambda img, l: ImageEnhance.Contrast(img).enhance(_flt(l, 1.8) + 0.1),
    lambda img, l: ImageEnhance.Brightness(img).enhance(_flt(l, 1.8) + 0.1),
    lambda img, l: ImageEnhance.Sharpness(img).enhance(_flt(l, 1.8) + 0.1),
    lambda img, l: gaussian_noise(img),
    lambda img, l: motion_blur(img),
    lambda img, l: pixelate(img),
]

def augmix_pil(image, severity=3, width=3, alpha=1.):
    ws  = np.float32(np.random.dirichlet([alpha] * width))
    m   = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(np.array(image), dtype=np.float32)
    for i in range(width):
        img_aug = image.copy()
        for _ in range(np.random.randint(1, 4)):
            img_aug = random.choice(AUG_OPS)(img_aug, np.random.randint(1, severity + 1))
        mix += ws[i] * np.array(img_aug, dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip((1-m)*np.array(image, np.float32) + m*mix, 0, 255)))

# ── 5. Transforms ─────────────────────────────────────────────────────────────
clean_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

augmix_base_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=CFG.randaug_n, magnitude=CFG.randaug_m),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
])

val_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def build_tta_transforms():
    tfms    = []
    to_norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    
    offsets = [(0,0),(0,8),(8,0),(8,8),(4,4),(2,6),(6,2),(0,4),(4,0),(4,8)]
    for flip in [False, True]:
        for r, c in offsets:
            ops = []
            if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
            ops.append(transforms.Pad(4, padding_mode='reflect'))
            ops.append(transforms.Lambda(lambda img,r=r,c=c: transforms.functional.crop(img,r,c,32,32)))
            ops += to_norm
            tfms.append(transforms.Compose(ops))

    jitter = transforms.ColorJitter(0.15, 0.15, 0.15)
    for flip in [False, True]:
        for r, c in offsets[:5]:
            ops = []
            if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
            ops += [jitter, transforms.Pad(4, padding_mode='reflect')]
            ops.append(transforms.Lambda(lambda img,r=r,c=c: transforms.functional.crop(img,r,c,32,32)))
            ops += to_norm
            tfms.append(transforms.Compose(ops))

    for flip in [False, True]:
        for r, c in offsets[:5]:
            ops = []
            if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
            def b_n(img):
                if random.random() < 0.5: img = img.filter(ImageFilter.BoxBlur(1))
                if random.random() < 0.5: img = gaussian_noise(img, (0.01, 0.04))
                return img
            ops.append(transforms.Lambda(b_n))
            ops.append(transforms.Pad(4, padding_mode='reflect'))
            ops.append(transforms.Lambda(lambda img,r=r,c=c: transforms.functional.crop(img,r,c,32,32)))
            ops += to_norm
            tfms.append(transforms.Compose(ops))

    for _ in range(10):
        ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda img: augmix_pil(img, severity=2, width=2)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        tfms.append(transforms.Compose(ops))

    return tfms[:CFG.tta_n]

TTA_TRANSFORMS = build_tta_transforms()

# ── 6. Datasets ────────────────────────────────────────────────────────────────
def _resolve_path(img_dir, id_):
    p = img_dir / f"{str(id_).zfill(6)}.png"
    if not p.exists():
        p = img_dir / f"{id_}.png"
    return p

class JSDDataset(Dataset):
    def __init__(self, df, img_dir, use_corruption=True):
        self.df   = df.reset_index(drop=True)
        self.dir  = img_dir
        self.use_c = use_corruption
        self.paths = [_resolve_path(img_dir, r['id']) for _, r in df.iterrows()]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.use_c:
            img = apply_corruption(img)
        x_clean = clean_tfm(img)
        x_aug   = augmix_base_tfm(augmix_pil(img))
        return x_clean, x_aug, int(self.df.iloc[idx]['label_idx'])

class CombinedJSDDatasetSoft(Dataset):
    """NEW: Returns 10D floating point targets. Converts original hard labels and handles soft pseudo labels."""
    def __init__(self, orig_df, orig_dir, pseudo_df, pseudo_dir, soft_dict=None):
        self.items = []
        for _, row in orig_df.iterrows():
            self.items.append((_resolve_path(orig_dir, row['id']), int(row['label_idx']), None))
        for _, row in pseudo_df.iterrows():
            self.items.append((_resolve_path(pseudo_dir, row['id']), int(row['label_idx']), str(row['id'])))
        self.soft_dict = soft_dict

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label, p_id = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = apply_corruption(img)
        x_clean = clean_tfm(img)
        x_aug   = augmix_base_tfm(augmix_pil(img))
        
        # Determine 10D target vector
        if self.soft_dict and p_id in self.soft_dict:
            target = self.soft_dict[p_id].clone()
        else:
            target = torch.zeros(CFG.num_classes)
            target[label] = 1.0
            
        return x_clean, x_aug, target

class ValDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df    = df.reset_index(drop=True)
        self.paths = [_resolve_path(img_dir, r['id']) for _, r in df.iterrows()]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return val_tfm(Image.open(self.paths[idx]).convert('RGB')), int(self.df.iloc[idx]['label_idx'])

class TestDataset(Dataset):
    def __init__(self, df, img_dir, tfm=None):
        self.df    = df.reset_index(drop=True)
        self.tfm   = tfm
        self.paths = [_resolve_path(img_dir, r['id']) for _, r in df.iterrows()]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.tfm: img = self.tfm(img)
        return img, str(self.df.iloc[idx]['id'])

# ── 7. Model ──────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep = 1 - self.p
        return x * (torch.rand(x.shape[0],1,1,1,device=x.device) < keep).float() / keep

class _WideBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, drop=0.3, drop_path=0.1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.drop  = drop; self.dp = DropPath(drop_path)
        self.skip  = None if (in_ch == out_ch and stride == 1) else \
                     nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)
    def forward(self, x):
        h   = F.relu(self.bn1(x))
        out = self.conv1(h)
        out = F.relu(self.bn2(out))
        if self.drop > 0 and self.training:
            out = F.dropout(out, p=self.drop, training=True)
        out = self.dp(self.conv2(out))
        return out + (self.skip(h) if self.skip else x)

class _WideGroup(nn.Module):
    def __init__(self, n, in_ch, out_ch, stride, drop=0.3, drop_path=0.1):
        super().__init__()
        layers = [_WideBlock(in_ch, out_ch, stride, drop, drop_path)]
        for _ in range(n - 1): layers.append(_WideBlock(out_ch, out_ch, 1, drop, drop_path))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, width=10, num_classes=10, drop=0.3, drop_path=0.1):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        chs = [16, 16*width, 32*width, 64*width]
        self.conv0  = nn.Conv2d(3, chs[0], 3, 1, 1, bias=False)
        self.group1 = _WideGroup(n, chs[0], chs[1], 1, drop, drop_path)
        self.group2 = _WideGroup(n, chs[1], chs[2], 2, drop, drop_path)
        self.group3 = _WideGroup(n, chs[2], chs[3], 2, drop, drop_path)
        self.bn     = nn.BatchNorm2d(chs[3])
        self.fc     = nn.Linear(chs[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv0(x)
        x = self.group1(x); x = self.group2(x); x = self.group3(x)
        x = F.relu(self.bn(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def build_model(load_ckpt=None):
    m = WideResNet(28, 10, CFG.num_classes, drop=0.3, drop_path=0.1).to(DEVICE)
    if load_ckpt:
        ckpt_path = Path(load_ckpt)
        if ckpt_path.exists():
            m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
            print(f"  ✅ Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  ⚠️  WARNING: Checkpoint not found at {ckpt_path}")
            print(f"  ⚠️  Training from scratch instead.")
    if USE_MULTI:
        m = nn.DataParallel(m)
        print(f"  DataParallel across {NUM_GPUS} GPUs")
    return m

# ── 8. Loss ────────────────────────────────────────────────────────────────────
class SoftCE(nn.Module):
    """Automatically parses hard (1D) or soft (2D) labels and applies smoothing."""
    def __init__(self, eps=0.1):
        super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        C = logits.size(1)
        if targets.dim() == 1: targets = F.one_hot(targets.long(), C).float()
        targets = (1 - self.eps) * targets + self.eps / C
        return -(targets * F.log_softmax(logits, 1)).sum(1).mean()

def jsd_loss(logits_clean, logits_aug):
    p_c   = F.softmax(logits_clean.float(), dim=1)
    p_a   = F.softmax(logits_aug.float(), dim=1)
    p_mix = (p_c + p_a) / 2.0
    return (F.kl_div(p_mix.log(), p_c, reduction='batchmean') +
            F.kl_div(p_mix.log(), p_a, reduction='batchmean')) / 2.0

criterion = SoftCE(CFG.label_smooth)

# ── 9. Mixup / CutMix ─────────────────────────────────────────────────────────
def rand_bbox(size, lam):
    H, W = size[-2], size[-1]
    r    = math.sqrt(1 - lam)
    h, w = int(H*r), int(W*r)
    cy, cx = random.randint(0, H), random.randint(0, W)
    return max(0,cx-w//2), max(0,cy-h//2), min(W,cx+w//2), min(H,cy+h//2)

def apply_mix(imgs, lbls):
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    if lbls.dim() == 1:
        oh = F.one_hot(lbls.long(), CFG.num_classes).float()
    else:
        oh = lbls  # Already soft targets from CombineJSDDatasetSoft

    if random.random() < 0.5:
        lam  = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
        imgs = lam * imgs + (1-lam) * imgs[perm]
    else:
        lam = np.random.beta(CFG.cutmix_alpha, CFG.cutmix_alpha)
        x1,y1,x2,y2 = rand_bbox(imgs.shape, lam)
        imgs = imgs.clone()
        imgs[:,:,y1:y2,x1:x2] = imgs[perm,:,y1:y2,x1:x2]
        lam = 1 - (x2-x1)*(y2-y1)/(imgs.shape[-1]*imgs.shape[-2])
        
    return imgs, lam*oh + (1-lam)*oh[perm]

def make_sampler(label_list):
    counts  = np.bincount(label_list, minlength=CFG.num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    sw      = torch.tensor([weights[l] for l in label_list], dtype=torch.float)
    return WeightedRandomSampler(sw, len(sw), replacement=True)

# ── 10. Soft Pseudo-labeling (NEW) ────────────────────────────────────────────
@torch.no_grad()
def generate_pseudo_labels(model, test_df):
    model.eval()
    print(f"\nGenerating SOFT pseudo-labels ({CFG.pseudo_tta} TTA passes, temp={CFG.pseudo_temp}) ...")

    all_probs = []
    for t in range(CFG.pseudo_tta):
        tta_tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: apply_corruption(img) if random.random() < 0.2 else img),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        ds = TestDataset(test_df, CFG.test_dir, tta_tfm)
        dl = DataLoader(ds, CFG.batch_size * 2, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True)
        probs_t = []
        for imgs, _ in dl:
            with autocast('cuda'):
                logits = model(imgs.to(DEVICE))
            probs_t.append(F.softmax(logits.float(), dim=1).cpu())
        all_probs.append(torch.cat(probs_t, dim=0))
        print(f"  pass {t+1}/{CFG.pseudo_tta}", end='\r')

    avg_probs = torch.stack(all_probs).mean(0)
    
    # Flatten the probability distribution to preserve uncertainty mathematically
    soft_probs = avg_probs ** (1 / CFG.pseudo_temp)
    soft_probs = soft_probs / soft_probs.sum(dim=1, keepdim=True)
    
    confs, preds = avg_probs.max(dim=1)
    
    accepted = confs >= CFG.pseudo_conf
    n_total  = len(test_df)
    n_accept = accepted.sum().item()

    if n_accept == 0:
        print(f"  ⚠️  0 samples at conf>{CFG.pseudo_conf}, falling back to top-40% by confidence")
        threshold_fallback = confs.kthvalue(int(0.6 * n_total)).values.item()
        accepted = confs >= threshold_fallback
        n_accept = accepted.sum().item()
        print(f"  Fallback threshold: {threshold_fallback:.4f}")

    print(f"\n  Accepted: {n_accept}/{n_total} ({100*n_accept/n_total:.1f}%) test samples")

    pseudo_rows = []
    soft_dict = {}
    for i, (id_val, accept, pred, conf) in enumerate(
            zip(test_df['id'], accepted, preds, confs)):
        if accept:
            pseudo_rows.append({
                'id': id_val, 'label': IDX2CLS[pred.item()],
                'label_idx': pred.item(), 'conf': conf.item()
            })
            soft_dict[str(id_val)] = soft_probs[i]

    if len(pseudo_rows) == 0:
        print("  ⚠️  No pseudo-labels generated — skipping pseudo-labeling phase")
        return None, None

    pseudo_df = pd.DataFrame(pseudo_rows).reset_index(drop=True)
    pseudo_df.to_csv(CFG.out_dir / 'pseudo_labels_soft.csv', index=False)
    torch.save(soft_dict, CFG.out_dir / 'pseudo_labels_soft.pt')
    
    print(f"  Class distribution (Hard Argmax approximation): {pseudo_df['label'].value_counts().to_dict()}")
    return pseudo_df, soft_dict

# ── 11. Train one epoch ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    run_ce = run_jsd = 0.0; nb = 0
    for x_clean, x_aug, lbls in loader:
        x_clean = x_clean.to(DEVICE, non_blocking=True)
        x_aug   = x_aug.to(DEVICE, non_blocking=True)
        lbls    = lbls.to(DEVICE, non_blocking=True)
        B = x_clean.size(0)
        
        if random.random() < CFG.mix_prob:
            x_clean, soft = apply_mix(x_clean, lbls)
        else:
            soft = lbls

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            all_logits   = model(torch.cat([x_clean, x_aug], dim=0))
            logits_clean = all_logits[:B]
            logits_aug   = all_logits[B:]
            loss_ce  = criterion(logits_clean, soft)
            loss_jsd = jsd_loss(logits_clean, logits_aug)
            loss     = loss_ce + CFG.lambda_jsd * loss_jsd
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params = model.module.parameters() if USE_MULTI else model.parameters()
        nn.utils.clip_grad_norm_(params, CFG.grad_clip)
        scaler.step(optimizer); scaler.update()
        
        run_ce += loss_ce.item(); run_jsd += loss_jsd.item(); nb += 1
    return run_ce / nb, run_jsd / nb

@torch.no_grad()
def validate(model, loader):
    model.eval(); preds, trues = [], []
    for imgs, lbls in loader:
        with autocast('cuda'):
            p = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
        preds.extend(p); trues.extend(lbls.numpy())
    return f1_score(trues, preds, average='macro'), np.mean(np.array(preds)==np.array(trues))

@torch.no_grad()
def predict_tta(model, df, img_dir):
    model.eval(); all_probs = []
    for i, tf in enumerate(TTA_TRANSFORMS, 1):
        ds = TestDataset(df, img_dir, tf)
        ld = DataLoader(ds, CFG.batch_size*2, num_workers=CFG.num_workers,
                        shuffle=False, pin_memory=True)
        probs = []
        for imgs, _ in ld:
            with autocast('cuda'):
                logits = model(imgs.to(DEVICE))
            probs.append(F.softmax(logits.float(), 1).cpu().numpy())
        all_probs.append(np.concatenate(probs, 0))
        print(f"    TTA {i}/{CFG.tta_n}", end='\r')
    print()
    return np.mean(all_probs, 0)

# ── 12. Data setup ────────────────────────────────────────────────────────────
print('Loading data ...')
train_df = pd.read_csv(CFG.train_csv)
train_df.columns = [c.strip().lower() for c in train_df.columns]
if   'label' in train_df.columns: train_df['label_idx'] = train_df['label'].map(CLS2IDX)
elif 'class' in train_df.columns: train_df['label_idx'] = train_df['class'].map(CLS2IDX)

test_df = pd.read_csv(CFG.sample_sub)
test_df.columns = [c.strip().lower() for c in test_df.columns]

tr_df, va_df = train_test_split(train_df, test_size=CFG.val_frac,
                                 stratify=train_df['label_idx'], random_state=CFG.seed)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
print(f"Train: {len(tr_df):,}  Val: {len(va_df):,}  Test: {len(test_df):,}\n")

va_ds = ValDataset(va_df, CFG.train_dir)
va_ld = DataLoader(va_ds, CFG.batch_size*4, shuffle=False,
                   num_workers=CFG.num_workers, pin_memory=True)

def make_train_loader(df, img_dir, pseudo_df=None, pseudo_dir=None, soft_dict=None):
    if pseudo_df is not None and len(pseudo_df) > 0:
        ds = CombinedJSDDatasetSoft(df, img_dir, pseudo_df, pseudo_dir, soft_dict)
        labels = list(df['label_idx']) + list(pseudo_df['label_idx'])
    else:
        ds = JSDDataset(df, img_dir, use_corruption=True)
        labels = list(df['label_idx'])
    sampler = make_sampler(labels)
    return DataLoader(ds, CFG.batch_size, sampler=sampler,
                      num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

# ── 13. Model / Optimizer / Scheduler ────────────────────────────────────────
print('Building model ...')
model = build_model(CFG.pretrained_ckpt)
raw   = model.module if USE_MULTI else model

optimizer = optim.SGD(raw.parameters(), lr=CFG.lr,
                      momentum=CFG.momentum, weight_decay=CFG.weight_decay,
                      nesterov=CFG.nesterov)
warmup_sched = optim.lr_scheduler.LambdaLR(
    optimizer, lambda ep: (ep+1)/CFG.warmup_eps if ep < CFG.warmup_eps else 1.0)
cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG.T0, T_mult=CFG.T_mult, eta_min=CFG.eta_min)

scaler    = GradScaler('cuda')
swa_model = AveragedModel(raw)
swa_sched = SWALR(optimizer, swa_lr=CFG.swa_lr, anneal_epochs=10)

# ── 14. Training Loop ─────────────────────────────────────────────────────────
best_f1   = 0.0
best_ckpt = CFG.out_dir / 'best_model_v5.pth'
log_rows  = []
t_total   = time.time()
pseudo_df = None
tr_ld     = make_train_loader(tr_df, CFG.train_dir)

print(f"\n{'Ep':>4} {'Phase':>7} {'CE':>7} {'JSD':>7} {'F1':>7} {'Acc':>7} {'LR':>9} {'Time':>6}")
print('─' * 65)

for epoch in range(1, CFG.epochs + 1):
    t0 = time.time()

    if epoch == CFG.pseudo_epoch + 1:
        pseudo_df, soft_dict = generate_pseudo_labels(model, test_df)
        if pseudo_df is not None:
            tr_ld = make_train_loader(tr_df, CFG.train_dir, pseudo_df, CFG.test_dir, soft_dict)
            print(f"  >> Phase 2 loader: {len(tr_ld.dataset):,} samples (Soft Target Mode ON)")
        else:
            print("  >> No pseudo-labels — continuing with original training data")

    phase = "SOFT_KD" if (pseudo_df is not None and epoch > CFG.pseudo_epoch) else "ORIG"
    loss_ce, loss_jsd = train_one_epoch(model, tr_ld, optimizer, scaler)

    if epoch <= CFG.warmup_eps:
        warmup_sched.step()
    elif epoch < CFG.swa_start:
        cosine_sched.step(epoch - CFG.warmup_eps)
    else:
        swa_model.update_parameters(raw)
        swa_sched.step()

    val_f1, val_acc = validate(model, va_ld)
    lr_now = optimizer.param_groups[0]['lr']

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(raw.state_dict(), best_ckpt)

    log_rows.append({'epoch': epoch, 'phase': phase, 'loss_ce': loss_ce,
                     'loss_jsd': loss_jsd, 'val_f1': val_f1, 'val_acc': val_acc, 'lr': lr_now})

    if epoch % 10 == 0 or epoch <= 5 or epoch == CFG.pseudo_epoch:
        star = ' ★' if val_f1 >= best_f1 else ''
        dt   = time.time() - t0
        print(f'{epoch:>4} {phase:>7} {loss_ce:>7.4f} {loss_jsd:>7.4f} {val_f1:>7.4f} '
              f'{val_acc*100:>6.2f}% {lr_now:>9.6f} {dt:>5.0f}s{star}')

elapsed = (time.time() - t_total) / 3600
print(f'\nTraining done in {elapsed:.2f} hrs  |  Best val F1: {best_f1:.4f}')

# ── 15. SWA finalization ──────────────────────────────────────────────────────
print('\nUpdating SWA BN stats ...')
class _SWAWrap:
    def __init__(self, ldr): self.ldr = ldr
    def __iter__(self):
        for x_c, x_a, y in self.ldr: yield x_c, y

update_bn(_SWAWrap(tr_ld), swa_model, device=DEVICE)
swa_f1, swa_acc = validate(swa_model, va_ld)
print(f'SWA F1: {swa_f1:.4f} | Best epoch F1: {best_f1:.4f}')

if swa_f1 >= best_f1:
    infer_model = swa_model; print('Using SWA model.')
else:
    raw.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=False))
    infer_model = raw; print(f'Using best checkpoint (F1={best_f1:.4f}).')

torch.save(raw.state_dict(), CFG.out_dir / 'final_model_v5.pth')
pd.DataFrame(log_rows).to_csv(CFG.out_dir / 'training_log_v5.csv', index=False)

# ── 16. TTA Inference + Submission ───────────────────────────────────────────
print(f'\nRunning TTA inference ({CFG.tta_n} passes) ...')
probs       = predict_tta(infer_model, test_df, CFG.test_dir)
pred_labels = [IDX2CLS[i] for i in probs.argmax(1)]

sub = pd.DataFrame({'id': test_df['id'], 'label': pred_labels})
sub.to_csv(CFG.out_dir / 'submission.csv', index=False)
print(f'\n✅  submission.csv saved  ({len(sub):,} rows)')
print(sub.head(10).to_string(index=False))
print('\nClass distribution:')
print(sub['label'].value_counts().to_string())
print(f'\nTotal time: {(time.time()-t_total)/3600:.2f} hrs')
