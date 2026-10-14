"""Fine-tune the continuous artwork decoder of QuantArt on a small artwork set.

Path (no codebook):  x -> encoder_real -> quant_conv_real -> post_quant_conv -> decoder -> x_hat
Frozen (eval mode):   encoder_real, quant_conv_real
Trained:              post_quant_conv, decoder (or only the modules given with --train_layers)
                      With --train_part encoder it is reversed: post_quant_conv + decoder are frozen
                      and encoder_real + quant_conv_real are trained (checkpoints: *_encoder.pt).
Loss:                 L1(x_hat, x) + perceptual_weight * LPIPS(x_hat, x), where x is the same
                      augmented image that was fed to the encoder (perceptual_weight=0: L1 only),
                      + optionally a PatchGAN hinge loss with taming's adaptive weight (--disc_weight > 0).
Data:                 random-resized crops: shorter side drawn from [size, size*scale_max] (capped at the
                      native resolution), so the decoder also sees brushstrokes near full resolution.
Optimisation:         Adam, linear warmup + cosine decay per step; an EMA copy of the decoder weights is
                      validated and saved (--ema 0 disables it).

Steps: baseline reconstructions on a few val images -> train with per-epoch validation
(best-val checkpoint on --select_metric, stop when train improves while val worsens) -> TensorBoard logs.
Test-set evaluation and the style-transfer comparison live in compare_decoders.py.
"""
import argparse, copy, json, math, os, pickle, random, time, types

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
    """train: shorter side -> random in [size, size*scale_max] (never above the native size),
    random size x size crop, horizontal flip with p=flip_p.
    val/test: shorter side -> size, center crop (fixed)."""

    def __init__(self, root, list_file, size=256, train=False, scale_max=1.125, flip_p=0.5):
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.size, self.train, self.flip_p = size, train, flip_p
        target = round(size * scale_max) if train else size
        # Decode and resize once to the largest scale needed and keep the images in memory.
        self.images = []
        for n in self.names:
            im = Image.open(os.path.join(root, n)).convert("RGB")
            self.images.append(resize_shorter(im, min(target, min(im.size)) if train else target))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im, s = self.images[i], self.size
        if self.train:
            short = min(im.size)
            if short > s:
                im = resize_shorter(im, random.randint(s, short))
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

    def encoder_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith(ENCODER_PREFIXES)}

    def freeze_decoder(self):
        for m in (self.post_quant_conv, self.decoder):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def encoder_parameters(self):
        return list(self.encoder_real.parameters()) + list(self.quant_conv_real.parameters())

    def set_train_mode(self, part):
        """Train mode only for the part being trained; the frozen part stays in eval mode."""
        trained = (self.encoder_real, self.quant_conv_real) if part == "encoder" else (self.post_quant_conv, self.decoder)
        self.eval()
        for m in trained:
            m.train()

    def freeze_encoder(self):
        for m in (self.encoder_real, self.quant_conv_real):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def decoder_parameters(self, train_layers=None):
        """Trainable decoder-side parameters. train_layers: module-name prefixes such as
        "decoder.up.0" or "decoder.conv_out"; everything else on the decoder side is frozen.
        None trains post_quant_conv and the whole decoder."""
        named = list(self.post_quant_conv.named_parameters(prefix="post_quant_conv")) + \
            list(self.decoder.named_parameters(prefix="decoder"))
        if train_layers is None:
            return [p for _, p in named]
        unknown = [l for l in train_layers if not any(n.startswith(l + ".") for n, _ in named)]
        if unknown:
            raise ValueError(f"no decoder parameters match {unknown}")
        params = []
        for n, p in named:
            train = any(n.startswith(l + ".") for l in train_layers)
            p.requires_grad_(train)
            if train:
                params.append(p)
        return params


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


# ----------------------------------------------------------------------------- GAN / EMA / schedule

def diff_augment(x, translation=0.125, cutout=0.5):
    """Differentiable translation + cutout (DiffAugment), applied to both real and fake images
    before the discriminator so it cannot simply memorise the 140 training artworks."""
    b, _, h, w = x.shape
    dy, dx = (random.randint(-int(s * translation), int(s * translation)) for s in (h, w))
    x = torch.roll(F.pad(x, (abs(dx), abs(dx), abs(dy), abs(dy))), shifts=(dy, dx), dims=(2, 3))
    x = x[:, :, abs(dy):abs(dy) + h, abs(dx):abs(dx) + w]
    ch, cw = int(h * cutout / 2), int(w * cutout / 2)
    ys = torch.randint(0, h - ch + 1, (b, 1, 1), device=x.device)
    xs = torch.randint(0, w - cw + 1, (b, 1, 1), device=x.device)
    gy = torch.arange(h, device=x.device).view(1, h, 1)
    gx = torch.arange(w, device=x.device).view(1, 1, w)
    hole = (gy >= ys) & (gy < ys + ch) & (gx >= xs) & (gx < xs + cw)
    return x * (~hole).unsqueeze(1).to(x.dtype)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def adaptive_gan_weight(rec_loss, g_loss, last_layer):
    # taming VQLPIPSWithDiscriminator.calculate_adaptive_weight: balance the gradient norms at the last layer.
    rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    return (rec_grad.norm() / (g_grad.norm() + 1e-4)).clamp(0.0, 1e4).detach()


@torch.no_grad()
def ema_update(ema_model, model, step, decay):
    # Warmup (as in timm/TF): short effective horizon early so the EMA does not lag the start.
    d = min(decay, (1 + step) / (10 + step))
    for (_, pe), (_, p) in zip(ema_model.named_parameters(), model.named_parameters()):
        if p.requires_grad:
            pe.lerp_(p, 1.0 - d)


def warmup_cosine(warmup_steps, total_steps, min_ratio):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return f


@torch.no_grad()
def detail(x):
    """Per-image high-frequency energy: mean |Laplacian| of the luminance."""
    y = x.mean(1, keepdim=True)
    k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=x.device).view(1, 1, 3, 3)
    return F.conv2d(y, k).abs().flatten(1).mean(1)


def evaluate(model, loader, device, lpips=None, perceptual_weight=0.0):
    model.eval()
    sums, n = {"loss": 0.0, "l1": 0.0, "lpips": 0.0, "psnr": 0.0, "sharpness": 0.0}, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = model.decode(model.encode(x))
        loss, l1, p = reconstruction_loss(x_hat, x, lpips, perceptual_weight)
        # sharpness: detail kept relative to the input (1 = as much texture as the painting; L1 blur pushes it down)
        sharp = detail(x_hat.clamp(-1, 1)) / detail(x).clamp_min(1e-6)
        for k, v in (("loss", loss), ("l1", l1), ("lpips", p), ("psnr", psnr(x_hat, x)), ("sharpness", sharp)):
            sums[k] += v.sum().item()
        n += x.shape[0]
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def reconstruct(model, x, device):
    model.eval()
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
    parser.add_argument("--decoder", default=None,
                        help="Optional exported decoder weights to load before training, e.g. a previously fine-tuned decoder")
    parser.add_argument("--data", default="datasets/CU_data_jpg")
    parser.add_argument("--split", default="datasets/CU_data_split")
    parser.add_argument("--out", default="outputs/finetune_decoder")
    parser.add_argument("--run", default=None, help="run name (default: timestamp)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--scale_max", type=float, default=2.5,
                        help="train crops: shorter side drawn from [size, size*scale_max] (1.125 ~ the old fixed 288 px)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="peak LR (config base_learning_rate 4.5e-6 x batch 4 = 1.8e-5 in taming)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="cosine decays to lr * this; 1 = constant LR")
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay of the trained weights (decoder or encoder side); 0 = off")
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--train_part", choices=["decoder", "encoder"], default="decoder",
                        help="decoder: freeze the encoder, train the decoder (default); "
                             "encoder: freeze post_quant_conv + decoder, train encoder_real + quant_conv_real")
    parser.add_argument("--train_layers", nargs="+", default=None,
                        help="decoder-side modules to train, e.g. decoder.up.0 decoder.conv_out; "
                             "all other decoder layers are frozen (default: post_quant_conv + whole decoder)")
    parser.add_argument("--perceptual_weight", type=float, default=1.0,
                        help="weight of the LPIPS term (taming VQGAN uses 1.0); 0 = L1 only")
    parser.add_argument("--disc_weight", type=float, default=0.0,
                        help="PatchGAN loss weight on top of taming's adaptive weight (taming: 1.0); 0 = no GAN")
    parser.add_argument("--disc_start", type=int, default=500,
                        help="step at which the GAN term enters the decoder loss (the discriminator trains from step 0)")
    parser.add_argument("--disc_layers", type=int, default=2, help="PatchGAN depth (taming: 3); smaller overfits less")
    parser.add_argument("--disc_lr", type=float, default=None, help="discriminator LR (default: --lr)")
    parser.add_argument("--select_metric", choices=["lpips", "loss", "l1"], default="lpips",
                        help="val metric for the best checkpoint and early stopping")
    parser.add_argument("--n_baseline", type=int, default=6, help="val images reconstructed as baseline")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many epochs where train loss improves but val is worse than best")
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

    train_set = ArtworkDataset(opt.data, os.path.join(opt.split, "train.txt"), opt.size, train=True,
                               scale_max=opt.scale_max, flip_p=opt.flip_p)
    val_set = ArtworkDataset(opt.data, os.path.join(opt.split, "val.txt"), opt.size)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.num_workers)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.num_workers)
    print(f"train {len(train_set)}  val {len(val_set)}")

    model = ContinuousArtworkAE.from_checkpoint(opt.config, opt.ckpt).to(device)
    if opt.decoder:
        decoder_sd = torch.load(opt.decoder, map_location="cpu", weights_only=True)
        if set(decoder_sd) != set(model.decoder_state_dict()):
            parser.error("--decoder must contain exactly post_quant_conv and decoder weights")
        model.load_state_dict(decoder_sd, strict=False)
        print(f"Loaded decoder: {opt.decoder}")
    if device.type == "mps":
        patch_for_mps(model.decoder)
        if opt.train_part == "encoder":
            patch_for_mps(model.encoder_real)
    if opt.train_part == "encoder":
        if opt.train_layers:
            parser.error("--train_layers selects decoder layers; it cannot be combined with --train_part encoder")
        model.freeze_decoder()
        params = model.encoder_parameters()
        print("train part: encoder_real + quant_conv_real (post_quant_conv + decoder frozen)")
    else:
        model.freeze_encoder()
        params = model.decoder_parameters(opt.train_layers)
        print(f"train part: decoder, layers: {opt.train_layers or 'post_quant_conv + whole decoder'}")
    n_train = sum(p.numel() for p in params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable {n_train/1e6:.2f}M  frozen {n_frozen/1e6:.2f}M")
    optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(0.5, 0.9))
    steps_per_epoch = len(train_loader) if not opt.max_steps else min(opt.max_steps, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(opt.warmup_steps, opt.epochs * steps_per_epoch, opt.min_lr_ratio))
    print(f"{steps_per_epoch} steps/epoch, {opt.epochs * steps_per_epoch} steps max, peak lr {opt.lr}")

    # The EMA copy holds the frozen part unchanged; validation, samples and checkpoints use it.
    ema_model = None
    if opt.ema > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
    eval_model = ema_model or model

    disc = None
    if opt.disc_weight > 0:
        from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
        disc = NLayerDiscriminator(input_nc=3, n_layers=opt.disc_layers).apply(weights_init).to(device)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=opt.disc_lr or opt.lr, betas=(0.5, 0.9))
        # Gradient balancing at the last trainable conv, as taming does at decoder.conv_out.
        last_layer = [p for p in params if p.ndim == 4][-1]
        print(f"PatchGAN: {opt.disc_layers} layers, weight {opt.disc_weight}, enters the loss at step {opt.disc_start}")

    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    # LPIPS is always computed on validation (a texture-aware quality metric);
    # it is part of the training loss only when --perceptual_weight > 0.
    lpips = load_lpips(device)
    if device.type == "mps":
        patch_for_mps(lpips)
    w = opt.perceptual_weight
    print(f"loss = L1 + {w} * LPIPS" + (f" + {opt.disc_weight} * adaptive * GAN" if disc else "")
          + f"   select on val {opt.select_metric}" + (f"   EMA {opt.ema}" if ema_model else ""))

    def log_val(metrics, epoch):
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    part = opt.train_part
    if part == "encoder":
        # Preserve the exact frozen decoder needed to reconstruct with encoder exports.
        torch.save(model.decoder_state_dict(), os.path.join(run_dir, "checkpoints", "frozen_decoder.pt"))

    def save_decoder(name):
        # Saves the trained part only: *_decoder.pt (decoder side) or *_encoder.pt (encoder side).
        sd = eval_model.encoder_state_dict() if part == "encoder" else eval_model.decoder_state_dict()
        torch.save(sd, os.path.join(run_dir, "checkpoints", name.replace("decoder", part)))

    # Baseline: pretrained autoencoder on a fixed set of val images.
    vis = [val_set[i] for i in range(min(opt.n_baseline, len(val_set)))]
    vis_x, vis_names = torch.stack([v[0] for v in vis]), [v[1] for v in vis]
    vis_base = reconstruct(model, vis_x, device)
    save_pairs(os.path.join(run_dir, "baseline"), vis_names, vis_x, vis_base)
    base = evaluate(model, val_loader, device, lpips, w)
    print(f"baseline  val loss {base['loss']:.4f}  L1 {base['l1']:.4f}  LPIPS {base['lpips']:.4f}  PSNR {base['psnr']:.2f}"
          f"  sharpness {base['sharpness']:.2f}")
    log_val(base, 0)
    writer.add_image("val/input_baseline", grid(vis_x, vis_base), 0)

    history = [{"epoch": 0, "train_loss": None, "train_l1": None, "train_lpips": None,
                **{f"val_{k}": v for k, v in base.items()}}]
    sel = opt.select_metric
    best_val, best_epoch, bad_epochs, prev_train = base[sel], 0, 0, None
    gan_selection_started = disc is None
    save_decoder("best_decoder.pt")

    def write_history():
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump({"best_epoch": best_epoch, "select_metric": sel, "best_val": best_val,
                       "history": history}, f, indent=2)

    write_history()  # --epochs 0 = baseline only: untouched weights saved, metrics recorded
    step = 0
    for epoch in range(1, opt.epochs + 1):
        model.set_train_mode(part)
        t0, sums, n = time.time(), {"loss": 0.0, "l1": 0.0, "lpips": 0.0}, 0
        for i, (x, _) in enumerate(train_loader):
            if opt.max_steps and i >= opt.max_steps:
                break
            x = x.to(device)
            if part == "encoder":
                h = model.encode(x)          # gradients flow back through the frozen decoder
            else:
                with torch.no_grad():
                    h = model.encode(x)
            x_hat = model.decode(h)
            # target = the same augmented image that went into the encoder
            rec, l1, p = reconstruction_loss(x_hat, x, lpips if w > 0 else None, w)
            rec = rec.mean()
            loss = rec
            if disc is not None and step >= opt.disc_start:
                disc.requires_grad_(False)
                g_loss = -disc(diff_augment(x_hat)).mean()
                d_weight = opt.disc_weight * adaptive_gan_weight(rec, g_loss, last_layer)  # encoder: quant_conv_real
                loss = rec + d_weight * g_loss
                disc.requires_grad_(True)
                writer.add_scalar("train/g_loss_step", g_loss.item(), step)
                writer.add_scalar("train/d_weight_step", d_weight.item(), step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                ema_update(ema_model, model, step, opt.ema)
            if disc is not None:
                logits_real = disc(diff_augment(x))
                logits_fake = disc(diff_augment(x_hat.detach()))
                d_loss = hinge_d_loss(logits_real, logits_fake)
                disc_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                disc_optimizer.step()
                writer.add_scalar("train/d_loss_step", d_loss.item(), step)
                writer.add_scalar("train/logits_real_step", logits_real.mean().item(), step)
                writer.add_scalar("train/logits_fake_step", logits_fake.mean().item(), step)
            step += 1
            # "loss" is the reconstruction objective (without the GAN term) so epochs stay comparable.
            for k, v in (("loss", rec.item()), ("l1", l1.mean().item()), ("lpips", p.mean().item())):
                sums[k] += v * x.shape[0]
            n += x.shape[0]
            writer.add_scalar("train/loss_step", rec.item(), step)
            writer.add_scalar("train/l1_step", l1.mean().item(), step)
            if w > 0:
                writer.add_scalar("train/lpips_step", p.mean().item(), step)
        train = {k: v / n for k, v in sums.items()}

        val = evaluate(eval_model, val_loader, device, lpips, w)
        vis_now = reconstruct(eval_model, vis_x, device)
        for k, v in train.items():
            if k != "lpips" or w > 0:
                writer.add_scalar(f"train/{k}", v, epoch)
        log_val(val, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_image("val/input_baseline_current", grid(vis_x, vis_base, vis_now), epoch)
        os.makedirs(os.path.join(run_dir, "val_samples"), exist_ok=True)
        torchvision.utils.save_image(grid(vis_x, vis_base, vis_now),
                                     os.path.join(run_dir, "val_samples", f"epoch_{epoch:02d}.png"))

        # Model selection and early stopping use --select_metric on validation (LPIPS by default:
        # loss values of runs with different weights are not comparable, and L1 rewards blur).
        # With the GAN on, only epochs that end after it entered the loss compete for best_*.pt, and early
        # stopping waits for it too: pre-GAN epochs usually win on LPIPS but are the blurry ones.
        gan_phase = disc is None or step > opt.disc_start
        if gan_phase and not gan_selection_started:
            gan_selection_started, best_val, bad_epochs = True, float("inf"), 0
        improved = gan_phase and val[sel] < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val[sel], epoch, 0
            save_decoder("best_decoder.pt")
        elif gan_phase and prev_train is not None and train["loss"] < prev_train:
            bad_epochs += 1
        prev_train = train["loss"]
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:2d}  train loss {train['loss']:.4f}  val loss {val['loss']:.4f}"
              f"  L1 {val['l1']:.4f}  LPIPS {val['lpips']:.4f}  PSNR {val['psnr']:.2f}  sharp {val['sharpness']:.2f}"
              f"  lr {optimizer.param_groups[0]['lr']:.2e}  {'*best*' if improved else ''}  {time.time()-t0:.0f}s")
        write_history()
        if part == "encoder":
            save_decoder("last_decoder.pt")
        if bad_epochs >= opt.patience:
            print(f"early stop: train improving while val worse than best for {bad_epochs} epochs")
            break

    save_decoder("last_decoder.pt")
    if ema_model is not None:
        raw = model.encoder_state_dict() if part == "encoder" else model.decoder_state_dict()
        torch.save(raw, os.path.join(run_dir, "checkpoints", f"last_{part}_raw.pt"))
    writer.close()
    print(f"best epoch {best_epoch}  val {sel} {best_val:.4f}  -> {run_dir}/checkpoints/best_{part}.pt")


if __name__ == "__main__":
    main()
