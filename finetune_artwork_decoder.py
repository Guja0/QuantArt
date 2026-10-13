"""Fine-tune the continuous artwork decoder of QuantArt on a small artwork set.

Path (no codebook):  x -> encoder_real -> quant_conv_real -> post_quant_conv -> decoder -> x_hat
Frozen (eval mode):   encoder_real, quant_conv_real
Trained:              post_quant_conv, decoder
Loss:                 L1(x_hat, x) + perceptual_weight * LPIPS(x_hat, x), where x is the same
                      augmented image that was fed to the encoder (perceptual_weight=0: L1 only).

Steps: baseline reconstructions on a few val images -> train with per-epoch validation
(best-val checkpoint, stop when train improves while val worsens) -> TensorBoard logs.
Test-set evaluation and the style-transfer comparison live in compare_decoders.py.
"""
import argparse, json, os, pickle, random, time, types

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from taming.modules.diffusionmodules.model import Decoder, Encoder

ENCODER_PREFIXES = ("encoder_real.", "quant_conv_real.")
DECODER_PREFIXES = ("post_quant_conv.", "decoder.")


# ----------------------------------------------------------------------------- data

class _LenientUnpickler(pickle.Unpickler):
    """Checkpoints also pickle pytorch-lightning callback objects; stub any class that
    cannot be imported so state_dict can be read without pytorch-lightning installed."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {"__setstate__": lambda self, state: None})


_lenient_pickle = types.ModuleType("lenient_pickle")
_lenient_pickle.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
_lenient_pickle.Unpickler = _LenientUnpickler


def load_state_dict(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", pickle_module=_lenient_pickle)["state_dict"]


def resize_shorter(im, size):
    w, h = im.size
    s = size / min(w, h)
    return im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def to_tensor(im):
    # Checkpoint normalization: RGB in [-1, 1] (x / 127.5 - 1), same as taming.data.base.
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
    return x.permute(2, 0, 1)


def to_image(x):
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).cpu().numpy())


class ArtworkDataset(Dataset):
    """train: shorter side -> size*scale, random size x size crop, horizontal flip with p=flip_p.
    val/test: shorter side -> size, center crop (fixed)."""

    def __init__(self, root, list_file, size=256, train=False, scale=1.125, flip_p=0.5):
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.size, self.train, self.flip_p = size, train, flip_p
        target = round(size * scale) if train else size
        # Images are small, so decode and resize once and keep them in memory.
        self.images = [resize_shorter(Image.open(os.path.join(root, n)).convert("RGB"), target) for n in self.names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im, s = self.images[i], self.size
        w, h = im.size
        if self.train:
            left, top = random.randint(0, w - s), random.randint(0, h - s)
        else:
            left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s))
        if self.train and random.random() < self.flip_p:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(im), self.names[i]


# ----------------------------------------------------------------------------- model

class ContinuousArtworkAE(torch.nn.Module):
    """The artwork autoencoder of a QuantArt checkpoint, without the codebook."""

    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder_real = Encoder(**ddconfig)
        self.quant_conv_real = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.decoder = Decoder(**ddconfig)

    @classmethod
    def from_checkpoint(cls, config_path, ckpt_path):
        p = OmegaConf.load(config_path).model.params
        model = cls(p.ddconfig, p.embed_dim)
        sd = load_state_dict(ckpt_path)
        sd = {k: v for k, v in sd.items() if k.startswith(ENCODER_PREFIXES + DECODER_PREFIXES)}
        model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x):
        return self.quant_conv_real(self.encoder_real(x))

    def decode(self, h):
        return self.decoder(self.post_quant_conv(h))

    def decoder_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(DECODER_PREFIXES)}

    def freeze_encoder(self):
        for m in (self.encoder_real, self.quant_conv_real):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def decoder_parameters(self):
        return list(self.post_quant_conv.parameters()) + list(self.decoder.parameters())


def _contiguous_grad_hook(module, inputs, output):
    if output.requires_grad:
        output.register_hook(lambda g: g.contiguous())


def patch_for_mps(model):
    # MPS conv2d backward fails on the non-contiguous gradients that the attention
    # block's permute/bmm produce; making conv-output gradients contiguous fixes it
    # (numerically identical to CPU, checked to ~1e-5).
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(_contiguous_grad_hook)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(x, y):
    # x, y in [-1, 1]; PSNR on the [0, 1] range, per image.
    mse = ((x.clamp(-1, 1) - y) / 2).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse)


def load_lpips(device):
    # Frozen VGG16-based LPIPS from taming (weights: torchvision VGG16 + taming vgg.pth).
    from taming.modules.losses.lpips import LPIPS
    return LPIPS().eval().to(device)


def reconstruction_loss(x_hat, x, lpips, perceptual_weight):
    """Per-image losses, as in the taming VQGAN: |x - x_hat| + w * LPIPS(x, x_hat)."""
    l1 = F.l1_loss(x_hat, x, reduction="none").flatten(1).mean(1)
    p = lpips(x_hat.contiguous(), x.contiguous()).flatten() if lpips is not None else torch.zeros_like(l1)
    return l1 + perceptual_weight * p, l1, p


@torch.no_grad()
def evaluate(model, loader, device, lpips=None, perceptual_weight=0.0):
    model.decoder.eval()
    sums, n = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "psnr": 0.0}, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = model.decode(model.encode(x))
        loss, l1, p = reconstruction_loss(x_hat, x, lpips, perceptual_weight)
        for k, v in (("loss", loss), ("l1", l1), ("lpips", p), ("psnr", psnr(x_hat, x))):
            sums[k] += v.sum().item()
        n += x.shape[0]
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def reconstruct(model, x, device):
    model.decoder.eval()
    return model.decode(model.encode(x.to(device))).cpu()


def save_pairs(out_dir, names, x, x_hat):
    for sub, batch in (("input", x), ("reconstructed", x_hat)):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        for n, img in zip(names, batch):
            to_image(img).save(os.path.join(out_dir, sub, os.path.splitext(n)[0] + ".png"))


def grid(*rows):
    # Each row: a batch in [-1, 1]; returns one image tensor in [0, 1] with rows stacked.
    x = torch.cat(rows)
    return torchvision.utils.make_grid((x.clamp(-1, 1) + 1) / 2, nrow=rows[0].shape[0], padding=2)


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="landscape2art/configs/test.yaml")
    parser.add_argument("--ckpt", default="landscape2art/checkpoints/last.ckpt")
    parser.add_argument("--data", default="datasets/CU_data_jpg")
    parser.add_argument("--split", default="datasets/CU_data_split")
    parser.add_argument("--out", default="outputs/finetune_decoder")
    parser.add_argument("--run", default=None, help="run name (default: timestamp)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--perceptual_weight", type=float, default=0.0,
                        help="weight of the LPIPS term (taming VQGAN uses 1.0); 0 = L1 only")
    parser.add_argument("--n_baseline", type=int, default=6, help="val images reconstructed as baseline")
    parser.add_argument("--patience", type=int, default=2,
                        help="stop after this many epochs where train loss improves but val loss is worse than best")
    parser.add_argument("--max_steps", type=int, default=None, help="debug: stop each epoch after this many steps")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    device = pick_device(opt.device)
    run_dir = os.path.join(opt.out, opt.run or time.strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(opt), f, indent=2)
    print(f"run dir: {run_dir}  device: {device}")

    train_set = ArtworkDataset(opt.data, os.path.join(opt.split, "train.txt"), opt.size, train=True, flip_p=opt.flip_p)
    val_set = ArtworkDataset(opt.data, os.path.join(opt.split, "val.txt"), opt.size)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_workers)
    print(f"train {len(train_set)}  val {len(val_set)}")

    model = ContinuousArtworkAE.from_checkpoint(opt.config, opt.ckpt).to(device)
    if device.type == "mps":
        patch_for_mps(model.decoder)
    model.freeze_encoder()
    params = model.decoder_parameters()
    n_train = sum(p.numel() for p in params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable {n_train/1e6:.1f}M  frozen {n_frozen/1e6:.1f}M")
    optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(0.5, 0.9))
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    # LPIPS is always computed on validation (a texture-aware quality metric);
    # it is part of the training loss only when --perceptual_weight > 0.
    lpips = load_lpips(device)
    if device.type == "mps":
        patch_for_mps(lpips)
    w = opt.perceptual_weight
    print(f"loss = L1 + {w} * LPIPS")

    def log_val(metrics, epoch):
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    # Baseline: pretrained decoder on a fixed set of val images.
    vis = [val_set[i] for i in range(min(opt.n_baseline, len(val_set)))]
    vis_x, vis_names = torch.stack([v[0] for v in vis]), [v[1] for v in vis]
    vis_base = reconstruct(model, vis_x, device)
    save_pairs(os.path.join(run_dir, "baseline"), vis_names, vis_x, vis_base)
    base = evaluate(model, val_loader, device, lpips, w)
    print(f"baseline  val loss {base['loss']:.4f}  L1 {base['l1']:.4f}  LPIPS {base['lpips']:.4f}  PSNR {base['psnr']:.2f}")
    log_val(base, 0)
    writer.add_image("val/input_baseline", grid(vis_x, vis_base), 0)

    history = [{"epoch": 0, "train_loss": None, "train_l1": None, "train_lpips": None,
                **{f"val_{k}": v for k, v in base.items()}}]
    best_val, best_epoch, bad_epochs, prev_train = base["loss"], 0, 0, None
    torch.save(model.decoder_state_dict(), os.path.join(run_dir, "checkpoints", "best_decoder.pt"))
    step = 0
    for epoch in range(1, opt.epochs + 1):
        model.decoder.train()
        model.post_quant_conv.train()
        t0, sums, n = time.time(), {"loss": 0.0, "l1": 0.0, "lpips": 0.0}, 0
        for i, (x, _) in enumerate(train_loader):
            if opt.max_steps and i >= opt.max_steps:
                break
            x = x.to(device)
            with torch.no_grad():
                h = model.encode(x)
            x_hat = model.decode(h)
            # target = the same augmented image that went into the encoder
            loss, l1, p = reconstruction_loss(x_hat, x, lpips if w > 0 else None, w)
            loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            for k, v in (("loss", loss.item()), ("l1", l1.mean().item()), ("lpips", p.mean().item())):
                sums[k] += v * x.shape[0]
            n += x.shape[0]
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/l1_step", l1.mean().item(), step)
            if w > 0:
                writer.add_scalar("train/lpips_step", p.mean().item(), step)
        train = {k: v / n for k, v in sums.items()}

        val = evaluate(model, val_loader, device, lpips, w)
        vis_now = reconstruct(model, vis_x, device)
        for k, v in train.items():
            if k != "lpips" or w > 0:
                writer.add_scalar(f"train/{k}", v, epoch)
        log_val(val, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_image("val/input_baseline_current", grid(vis_x, vis_base, vis_now), epoch)
        os.makedirs(os.path.join(run_dir, "val_samples"), exist_ok=True)
        torchvision.utils.save_image(grid(vis_x, vis_base, vis_now),
                                     os.path.join(run_dir, "val_samples", f"epoch_{epoch:02d}.png"))

        # Model selection and early stopping use the training objective on validation.
        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val["loss"], epoch, 0
            torch.save(model.decoder_state_dict(), os.path.join(run_dir, "checkpoints", "best_decoder.pt"))
        elif prev_train is not None and train["loss"] < prev_train:
            bad_epochs += 1
        prev_train = train["loss"]
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:2d}  train loss {train['loss']:.4f}  val loss {val['loss']:.4f}"
              f"  L1 {val['l1']:.4f}  LPIPS {val['lpips']:.4f}  PSNR {val['psnr']:.2f}"
              f"  {'*best*' if improved else ''}  {time.time()-t0:.0f}s")
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "best_val_loss": best_val, "history": history}, f, indent=2)
        if bad_epochs >= opt.patience:
            print(f"early stop: train improving while val worse than best for {bad_epochs} epochs")
            break

    torch.save(model.decoder_state_dict(), os.path.join(run_dir, "checkpoints", "last_decoder.pt"))
    writer.close()
    print(f"best epoch {best_epoch}  val loss {best_val:.4f}  -> {run_dir}/checkpoints/best_decoder.pt")


if __name__ == "__main__":
    main()
