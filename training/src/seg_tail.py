"""
Tail Segmentation -- segmentation CASA pipeline.

Two-phase approach:
  Phase A  Bootstrap  -- classical Frangi/Sato ridge filter extracts thin
                         flagellar strands; each strand is linked to the nearest
                         Cellpose head centroid.  Produces weak-but-free labels
                         for the whole dataset.
  Phase B  U-Net      -- a lightweight encoder-decoder (ResNet-34 backbone via
                         segmentation-models-pytorch) trained on the bootstrapped
                         labels (optionally refined by hand).  Run inference
                         with the trained U-Net on every frame.

Outputs (under segmentation.tail_masks_dir):
    <video_id>_tails_bootstrap.npz  -- per-frame uint8 binary tail masks (Phase A)
    <video_id>_tails.npz            -- per-frame uint8 binary tail masks (Phase B)
    <video_id>_tails.json           -- per-frame per-sperm tail pixel lists (for beat analysis)

Usage:
    # Phase A: bootstrap tail masks from ridge filter
    python -m src.seg_tail --config configs/default.yaml --phase bootstrap
    python -m src.seg_tail --config configs/default.yaml --phase bootstrap --video-id 1

    # Phase B: train U-Net on bootstrapped labels
    python -m src.seg_tail --config configs/default.yaml --phase train

    # Phase B: run U-Net inference
    python -m src.seg_tail --config configs/default.yaml --phase infer
    python -m src.seg_tail --config configs/default.yaml --phase infer --video-id 1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(tail_cfg: dict | None = None) -> "torch.device":
    """Resolve PyTorch compute device (GPU if available/requested, else CPU)."""
    import torch
    dev_str = "cuda"
    if tail_cfg and isinstance(tail_cfg, dict):
        dev_str = str(tail_cfg.get("device", "cuda")).lower()

    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("  [!] CUDA device requested but PyTorch CUDA is unavailable. Using CPU fallback.")
            return torch.device("cpu")
        torch.backends.cudnn.benchmark = True
        return torch.device(dev_str)
    return torch.device("cpu")


def _resolve_dir(path_str: str, config_path: str) -> Path:
    """Resolve a directory path relative to CWD or config file location."""
    p = Path(path_str)
    if p.is_absolute() or p.exists():
        return p
    cfg_dir = Path(config_path).resolve().parent
    cand1 = cfg_dir.parent / p
    if cand1.exists():
        return cand1
    cand2 = cfg_dir / p
    if cand2.exists():
        return cand2
    return p


def _find_weights(out_dir: Path, config_path: str) -> Path:
    """Find unet_tail.pt weights file across common output locations."""
    w = out_dir / "unet_tail.pt"
    if w.exists():
        return w
    cfg_dir = Path(config_path).resolve().parent
    cands = [
        cfg_dir.parent / "outputs" / "seg" / "tail" / "unet_tail.pt",
        cfg_dir.parent / "training" / "outputs" / "seg" / "tail" / "unet_tail.pt",
        cfg_dir / "outputs" / "seg" / "tail" / "unet_tail.pt",
    ]
    for c in cands:
        if c.exists():
            return c
    return w


def _load_head_centroids(heads_json: Path) -> dict[str, list[list[float]]]:
    """Return {frame_str: [[cx, cy], ...]} from a _heads.json file."""
    if not heads_json.exists():
        return {}
    with open(heads_json) as f:
        d = json.load(f)
    return {k: [h["centroid"] for h in v] for k, v in d.get("frames", {}).items()}


def _frame_index(stem: str, fallback: int) -> int:
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return fallback


def _only_valid_frames(frame_files: list[Path], frames_dir: Path, video_id: str, cfg: dict) -> list[Path]:
    """Drop black-gap and frozen-still frames (src/frame_qc.py).  Tracks only exist
    on live frames, and duplicated stills would over-weight U-Net training."""
    from src.frame_qc import load_valid_frames
    valid = load_valid_frames(frames_dir, video_id, cfg)
    if valid is None:
        return frame_files
    return [ff for ff in frame_files if _frame_index(ff.stem, -1) in valid]


# ---------------------------------------------------------------------------
# Phase A: Ridge-filter bootstrap
# ---------------------------------------------------------------------------

def _ridge_mask_gpu(gray: np.ndarray, sigmas: list[float], threshold: float, device: "torch.device") -> np.ndarray:
    """GPU-accelerated Frangi ridge filter using PyTorch 2D convolutions."""
    import torch
    import torch.nn.functional as F

    img_t = torch.from_numpy(gray).float().to(device) / 255.0
    inv_t = 1.0 - img_t.unsqueeze(0).unsqueeze(0)
    max_response = torch.zeros_like(inv_t)

    for sigma in sigmas:
        radius = max(1, int(round(3.0 * sigma)))
        x = torch.arange(-radius, radius + 1, device=device).float()
        g = torch.exp(-x**2 / (2 * sigma**2))
        g = g / g.sum()
        d1g = -x / (sigma**2) * g
        d2g = (x**2 / (sigma**4) - 1.0 / (sigma**2)) * g

        g_col, g_row = g.view(1, 1, -1, 1), g.view(1, 1, 1, -1)
        d1g_col, d1g_row = d1g.view(1, 1, -1, 1), d1g.view(1, 1, 1, -1)
        d2g_col, d2g_row = d2g.view(1, 1, -1, 1), d2g.view(1, 1, 1, -1)

        padded = F.pad(inv_t, (radius, radius, radius, radius), mode="reflect")
        ixx = F.conv2d(F.conv2d(padded, d2g_row), g_col) * (sigma**2)
        iyy = F.conv2d(F.conv2d(padded, g_row), d2g_col) * (sigma**2)
        ixy = F.conv2d(F.conv2d(padded, d1g_row), d1g_col) * (sigma**2)

        tmp_diff = ixx - iyy
        tmp_sq = torch.sqrt(tmp_diff**2 + 4 * ixy**2)
        l1 = 0.5 * (ixx + iyy + tmp_sq)
        l2 = 0.5 * (ixx + iyy - tmp_sq)

        abs_l1, abs_l2 = torch.abs(l1), torch.abs(l2)
        e1 = torch.where(abs_l1 <= abs_l2, l1, l2)
        e2 = torch.where(abs_l1 <= abs_l2, l2, l1)

        rb_sq = (e1 / (e2 + 1e-10))**2
        s_sq = e1**2 + e2**2
        vesselness = torch.exp(-rb_sq / 0.5) * (1.0 - torch.exp(-s_sq / 0.5))
        vesselness = torch.where(e2 < 0, vesselness, torch.zeros_like(vesselness))
        max_response = torch.maximum(max_response, vesselness)

    rmin, rmax = max_response.min(), max_response.max()
    if rmax > rmin:
        max_response = (max_response - rmin) / (rmax - rmin)

    return (max_response[0, 0] > threshold).to(torch.uint8).cpu().numpy() * 255


def _ridge_mask(
    gray: np.ndarray,
    filter_name: str,
    sigmas: list[float],
    threshold: float,
    device: "torch.device" | None = None,
) -> np.ndarray:
    """Run a ridge/vesselness filter (GPU if available, else CPU) and return uint8 binary mask."""
    if device is not None and device.type == "cuda" and filter_name == "frangi":
        try:
            return _ridge_mask_gpu(gray, sigmas, threshold, device)
        except Exception:
            pass  # Fallback to CPU skimage if GPU execution fails

    from skimage.filters import frangi, sato, meijering  # type: ignore

    img_float = gray.astype(np.float32) / 255.0
    inv = 1.0 - img_float
    fn = {"frangi": frangi, "sato": sato, "meijering": meijering}.get(filter_name, frangi)
    ridge = fn(inv, sigmas=sigmas, black_ridges=False)

    rmin, rmax = ridge.min(), ridge.max()
    if rmax > rmin:
        ridge = (ridge - rmin) / (rmax - rmin)

    return (ridge > threshold).astype(np.uint8) * 255


def _link_tails_to_heads(
    tail_mask: np.ndarray,
    centroids: list[list[float]],
    max_dist_px: float,
) -> dict[int, np.ndarray]:
    """
    Label connected components of the tail binary mask and assign each
    component to the nearest head centroid (if within max_dist_px).

    Returns {head_index: component_mask_uint8}.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required.")
    if not centroids or tail_mask.max() == 0:
        return {}

    n_labels, labels_cc, stats, centroids_cc = cv2.connectedComponentsWithStats(
        tail_mask, connectivity=8
    )

    head_arr = np.array(centroids)  # (N, 2) [cx, cy]
    assignments: dict[int, np.ndarray] = {}

    for cc_id in range(1, n_labels):
        # Centroid of this tail component
        cx_cc, cy_cc = centroids_cc[cc_id]
        dists = np.sqrt(((head_arr[:, 0] - cx_cc) ** 2) + ((head_arr[:, 1] - cy_cc) ** 2))
        nearest_head = int(dists.argmin())
        if dists[nearest_head] > max_dist_px:
            continue  # too far from any head -- debris / noise

        component_mask = (labels_cc == cc_id).astype(np.uint8) * 255
        if nearest_head in assignments:
            # Merge additional components linked to the same head
            assignments[nearest_head] = cv2.bitwise_or(assignments[nearest_head], component_mask)
        else:
            assignments[nearest_head] = component_mask

    return assignments


def bootstrap_tails(config_path: str = "configs/default.yaml",
                    video_id: str | None = None) -> None:
    """Phase A: ridge-filter bootstrap for all (or one) video."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg = cfg.get("segmentation", {})
    tail_cfg = seg_cfg.get("tail", {})

    frames_dir = _resolve_dir(paths_cfg.get("frames_dir", "data/frames"), config_path)
    head_dir = _resolve_dir(seg_cfg.get("head_masks_dir", "outputs/seg/head"), config_path)
    out_dir = _resolve_dir(seg_cfg.get("tail_masks_dir", "outputs/seg/tail"), config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_interval = int(seg_cfg.get("sample_interval", 1))
    filter_name = tail_cfg.get("ridge_filter", "frangi")
    sigmas = tail_cfg.get("ridge_sigmas", [1, 2, 3])
    threshold = float(tail_cfg.get("ridge_threshold", 0.15))
    max_link = float(tail_cfg.get("max_link_dist_px", 40))

    if cv2 is None:
        raise ImportError("opencv-python is required.")

    try:
        import skimage  # noqa: F401
    except ImportError as e:
        raise ImportError("scikit-image required. pip install scikit-image") from e

    device = _get_device(tail_cfg)
    frame_dirs = [frames_dir / str(video_id)] if video_id else \
        sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    print("=== Tail Bootstrap (ridge filter) ===")
    print(f"Filter: {filter_name} | sigmas: {sigmas} | threshold: {threshold} | max_link: {max_link}px | device: {device}")

    for fdir in frame_dirs:
        if not fdir.is_dir():
            continue
        vid = fdir.name
        heads_json = head_dir / f"{vid}_heads.json"
        centroids_by_frame = _load_head_centroids(heads_json)

        frame_files = sorted(fdir.glob("*.jpg"))[::sample_interval]
        frame_files = _only_valid_frames(frame_files, frames_dir, vid, cfg)
        if not frame_files:
            print(f"  [!] no frames in {fdir}")
            continue

        print(f"  Video {vid}: {len(frame_files)} frame(s)...")
        mask_store: dict[str, np.ndarray] = {}
        per_frame_json: dict[str, list] = {}
        total_tails = 0

        for i, ff in enumerate(frame_files):
            img = cv2.imread(str(ff), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            fidx = _frame_index(ff.stem, i)
            ridge = _ridge_mask(img, filter_name, sigmas, threshold, device=device)
            centroids = centroids_by_frame.get(str(fidx), [])
            assignments = _link_tails_to_heads(ridge, centroids, max_link)

            # Combine all assigned tail masks into one frame mask for storage
            combined = np.zeros_like(img, dtype=np.uint8)
            frame_tails = []
            for head_idx, tmask in assignments.items():
                combined = cv2.bitwise_or(combined, tmask)
                ys, xs = np.where(tmask > 0)
                frame_tails.append({
                    "head_idx": head_idx,
                    "n_pixels": int(xs.size),
                    "centroid": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)] if xs.size else None,
                })
            mask_store[ff.stem] = combined
            per_frame_json[str(fidx)] = frame_tails
            total_tails += len(assignments)

        npz_path = out_dir / f"{vid}_tails_bootstrap.npz"
        np.savez_compressed(npz_path, **mask_store)
        json_path = out_dir / f"{vid}_tails.json"
        with open(json_path, "w") as jf:
            json.dump({"video_id": vid, "phase": "bootstrap", "frames": per_frame_json}, jf, indent=2)
        print(f"    -> {total_tails} linked tails | {npz_path.name}")

    print("Bootstrap complete. Review outputs/seg/tail/ before U-Net training.")


# ---------------------------------------------------------------------------
# Phase B: U-Net training
# ---------------------------------------------------------------------------

def _build_unet(encoder: str, device):
    """Build a binary-segmentation U-Net from segmentation_models_pytorch."""
    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except ImportError as e:
        raise ImportError("segmentation-models-pytorch required. pip install segmentation-models-pytorch") from e
    import torch
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=1,  # grayscale
        classes=1,
        activation=None,
    ).to(device)
    return model


class _TailDataset:
    """
    Torch Dataset: pairs of (gray frame, binary tail mask), one pair per frame.

    RC-6 FIX: The original implementation loaded ONE representative image per
    video and reused it for every mask in that video's NPZ file.  This meant
    the U-Net was trained on completely mismatched pixel/mask pairs (e.g. frame
    0's pixels paired with frame 100's mask).  This caused tail detection to
    fail on frames that differed from frame 0, leading to high skip rates in
    beat analysis and degraded ALH/BCF estimates.

    This version correctly pairs each frame stem from the NPZ with the actual
    image file for that frame.
    """

    def __init__(
        self,
        frame_paths: list[Path],       # per-frame image paths
        mask_arrays: list[np.ndarray], # corresponding mask arrays (H, W) uint8
        img_size: int = 512,
        augment: bool = True,
    ):
        import torch
        self.pairs: list[tuple[Path, np.ndarray]] = list(zip(frame_paths, mask_arrays))
        self.img_size = img_size
        self.augment  = augment
        self.torch    = torch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, msk = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: blank frame (should not happen in practice)
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        img = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        msk = cv2.resize(msk, (self.img_size, self.img_size)).astype(np.float32) / 255.0

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1].copy()
                msk = msk[:, ::-1].copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = img[::-1, :].copy()
                msk = msk[::-1, :].copy()
            # Random 90-degree rotation (k in {0,1,2,3})
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()
                msk = np.rot90(msk, k).copy()
            # Random brightness jitter (microscopy contrast shifts)
            alpha = np.random.uniform(0.8, 1.2)
            img = np.clip(img * alpha, 0.0, 1.0).astype(np.float32)

        img_t = self.torch.tensor(img).unsqueeze(0)   # (1, H, W)
        msk_t = self.torch.tensor(msk).unsqueeze(0)   # (1, H, W)
        return img_t, msk_t


def _dice_bce_loss(logits: "torch.Tensor", targets: "torch.Tensor",
                   bce_weight: float = 0.5) -> "torch.Tensor":
    """
    Combined Dice + BCE loss.

    Pure BCE optimises pixel-wise accuracy, which on thin flagella (<<1% of
    pixels) means the model learns to predict all-background and still gets
    >99% pixel accuracy.  Dice loss is class-balanced and directly optimises
    the overlap of foreground pixels — critical for thin-structure segmentation.

    bce_weight=0.5 → equal contribution; raise toward 1.0 to trust BCE more.
    """
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce_weight * bce + (1.0 - bce_weight) * dice.mean()


def _iou_score(logits: "torch.Tensor", targets: "torch.Tensor",
               threshold: float = 0.5) -> float:
    """Compute mean IoU over a batch (used for early stopping)."""
    import torch
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(iou.mean().item())


def train_unet(config_path: str = "configs/default.yaml") -> None:
    """Phase B-train: train U-Net on bootstrapped tail masks."""
    import torch
    from torch.utils.data import DataLoader

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg   = cfg.get("segmentation", {})
    tail_cfg  = seg_cfg.get("tail", {})

    frames_dir  = Path(paths_cfg.get("frames_dir", "data/frames"))
    out_dir     = Path(seg_cfg.get("tail_masks_dir", "outputs/seg/tail"))
    weights_path = out_dir / "unet_tail.pt"

    encoder    = tail_cfg.get("unet_encoder",   "resnet34")
    epochs     = int(tail_cfg.get("unet_epochs",     60))
    batch_size = int(tail_cfg.get("unet_batch_size",  8))
    lr         = float(tail_cfg.get("unet_lr",      3e-4))
    patience   = int(tail_cfg.get("early_stop_patience", 12))  # epochs without IoU improvement

    from src.frame_qc import load_valid_frames

    device = _get_device(tail_cfg)
    print(f"=== U-Net Tail Training === encoder={encoder} | device={device} | epochs={epochs}")

    # ------------------------------------------------------------------
    # Build per-frame (image, mask) pairs — RC-6 FIX
    # ------------------------------------------------------------------
    frame_paths: list[Path]       = []
    mask_arrays: list[np.ndarray] = []

    for vid_dir in sorted([d for d in frames_dir.iterdir() if d.is_dir()]):
        npz_path = out_dir / f"{vid_dir.name}_tails_bootstrap.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        valid = load_valid_frames(frames_dir, vid_dir.name, cfg)
        for stem in z.files:
            if valid is not None and _frame_index(stem, -1) not in valid:
                continue                    # stale bootstrap from before frame QC
            # stem is like "frame_000123" — match the actual jpg on disk
            ff = vid_dir / f"{stem}.jpg"
            if ff.exists():
                frame_paths.append(ff)
                mask_arrays.append(z[stem])
            else:
                # Fallback: try stem without leading zeros (e.g. "frame_1")
                found = list(vid_dir.glob(f"{stem}*.jpg"))
                if found:
                    frame_paths.append(found[0])
                    mask_arrays.append(z[stem])

    if not frame_paths:
        print("[!] No per-frame bootstrap masks found. Run `--phase bootstrap` first.")
        return

    print(f"  Dataset: {len(frame_paths)} frame-mask pairs from {frames_dir}")

    # AUGMENTATION-LEAK FIX: random_split() returns two Subset objects that
    # both wrap the SAME underlying dataset instance, so mutating
    # val_ds.dataset.augment also mutated train_ds.dataset.augment (they are
    # literally the same object) — this silently disabled augmentation for
    # training as well. Split the raw index list instead and build two
    # independent _TailDataset instances so each split owns its own
    # `augment` flag. torch.randperm with the same seed reproduces exactly
    # what random_split does internally, so the split itself is unchanged.
    n_val   = max(1, int(0.15 * len(frame_paths)))
    n_train = len(frame_paths) - n_val
    perm = torch.randperm(len(frame_paths), generator=torch.Generator().manual_seed(42)).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = _TailDataset(
        [frame_paths[i] for i in train_idx], [mask_arrays[i] for i in train_idx], augment=True,
    )
    val_ds = _TailDataset(
        [frame_paths[i] for i in val_idx], [mask_arrays[i] for i in val_idx], augment=False,
    )

    pin_mem  = (device.type == "cuda")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model     = _build_unet(encoder, device)
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp   = (device.type == "cuda")
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_iou    = -1.0
    epochs_no_improve = 0

    print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val IoU':>8}")
    print(f"  {'------':>6}  {'-----------':>11}  {'---------':>9}  {'--------':>8}")

    for ep in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        t_loss = 0.0
        for imgs, msks in train_dl:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = _dice_bce_loss(model(imgs), msks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            t_loss += loss.item()
        scheduler.step()

        # ---------- validate ----------
        model.eval()
        v_loss = 0.0
        v_iou  = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    v_loss += _dice_bce_loss(logits, msks).item()
                    v_iou  += _iou_score(logits, msks)
                n_val_batches += 1

        t_loss /= max(len(train_dl), 1)
        v_loss /= max(n_val_batches, 1)
        v_iou  /= max(n_val_batches, 1)

        if ep % 5 == 0 or ep == 1:
            print(f"  {ep:>6}  {t_loss:>11.4f}  {v_loss:>9.4f}  {v_iou:>8.4f}")

        # ---------- checkpoint on best val IoU ----------
        if v_iou > best_val_iou:
            best_val_iou = v_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stop at epoch {ep} (no IoU improvement for {patience} epochs)")
                break

    print(f"Best val IoU {best_val_iou:.4f} | saved {weights_path}")


# ---------------------------------------------------------------------------
# Phase B: U-Net inference
# ---------------------------------------------------------------------------

def infer_unet(config_path: str = "configs/default.yaml",
               video_id: str | None = None) -> None:
    """Phase B-infer: run trained U-Net to produce refined tail masks."""
    import torch

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg = cfg.get("segmentation", {})
    tail_cfg = seg_cfg.get("tail", {})

    frames_dir = _resolve_dir(paths_cfg.get("frames_dir", "data/frames"), config_path)
    head_dir = _resolve_dir(seg_cfg.get("head_masks_dir", "outputs/seg/head"), config_path)
    out_dir = _resolve_dir(seg_cfg.get("tail_masks_dir", "outputs/seg/tail"), config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = _find_weights(out_dir, config_path)

    if not weights_path.exists():
        print(f"[!] No trained weights at {weights_path}. Run `--phase train` first.")
        return

    encoder = tail_cfg.get("unet_encoder", "resnet34")
    max_link = float(tail_cfg.get("max_link_dist_px", 40))
    sample_interval = int(seg_cfg.get("sample_interval", 1))
    batch_size = int(tail_cfg.get("unet_batch_size", 8))

    device = _get_device(tail_cfg)
    model = _build_unet(encoder, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    use_amp = (device.type == "cuda")
    print(f"=== U-Net Tail Inference === device={device} (AMP={use_amp})")

    frame_dirs = [frames_dir / str(video_id)] if video_id else \
        sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    for fdir in frame_dirs:
        if not fdir.is_dir():
            continue
        vid = fdir.name
        heads_json = head_dir / f"{vid}_heads.json"
        centroids_by_frame = _load_head_centroids(heads_json)
        frame_files = sorted(fdir.glob("*.jpg"))[::sample_interval]
        frame_files = _only_valid_frames(frame_files, frames_dir, vid, cfg)
        if not frame_files:
            continue

        print(f"  Video {vid}: {len(frame_files)} frame(s)...")
        mask_store: dict[str, np.ndarray] = {}
        per_frame_json: dict[str, list] = {}

        # Batched GPU inference loop
        for b_start in range(0, len(frame_files), batch_size):
            batch_files = frame_files[b_start : b_start + batch_size]
            batch_imgs = []
            valid_items = []

            for i, ff in enumerate(batch_files):
                img_bgr = cv2.imread(str(ff))
                if img_bgr is None:
                    continue
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                H, W = gray.shape
                inp = cv2.resize(gray, (512, 512)).astype(np.float32) / 255.0
                batch_imgs.append(inp)
                valid_items.append((b_start + i, ff, gray, H, W))

            if not batch_imgs:
                continue

            t_batch = torch.tensor(np.array(batch_imgs)).unsqueeze(1).to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                preds = torch.sigmoid(model(t_batch))[:, 0].float().cpu().numpy()

            for idx, (i_idx, ff, gray, H, W) in enumerate(valid_items):
                pred_full = cv2.resize(preds[idx], (W, H))
                binary = (pred_full > 0.5).astype(np.uint8) * 255

                fidx = _frame_index(ff.stem, i_idx)
                centroids = centroids_by_frame.get(str(fidx), [])
                assignments = _link_tails_to_heads(binary, centroids, max_link)

                combined = np.zeros_like(gray, dtype=np.uint8)
                frame_tails = []
                for head_idx, tmask in assignments.items():
                    combined = cv2.bitwise_or(combined, tmask)
                    ys, xs = np.where(tmask > 0)
                    frame_tails.append({
                        "head_idx": head_idx,
                        "n_pixels": int(xs.size),
                        "centroid": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)] if xs.size else None,
                    })
                mask_store[ff.stem] = combined
                per_frame_json[str(fidx)] = frame_tails

        npz_path = out_dir / f"{vid}_tails.npz"
        np.savez_compressed(npz_path, **mask_store)
        json_path = out_dir / f"{vid}_tails.json"
        with open(json_path, "w") as jf:
            json.dump({"video_id": vid, "phase": "unet", "frames": per_frame_json}, jf, indent=2)
        print(f"    -> saved {npz_path.name} + {json_path.name}")

    print("U-Net inference complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tail segmentation (bootstrap / U-Net train / U-Net infer)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", choices=["bootstrap", "train", "infer"], required=True,
                        help="bootstrap=ridge labels | train=fit U-Net | infer=run U-Net")
    parser.add_argument("--video-id", default=None, help="Single video id (e.g. '1')")
    args = parser.parse_args()

    if args.phase == "bootstrap":
        bootstrap_tails(args.config, args.video_id)
    elif args.phase == "train":
        train_unet(args.config)
    elif args.phase == "infer":
        infer_unet(args.config, args.video_id)


if __name__ == "__main__":
    main()
