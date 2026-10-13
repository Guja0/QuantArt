"""Reconstruct images with a single autoencoder, with and without vector quantization.

Default: the artwork (style) autoencoder inside a QuantArt stage-2 checkpoint,
encoder_real -> quant_conv_real -> [quantize_dec] -> post_quant_conv -> decoder.
With --stage1: a standalone stage-1 VQGAN checkpoint (e.g. the photo autoencoder
vqgan_landscape_f16_1024.ckpt), encoder -> quant_conv -> [quantize] -> post_quant_conv -> decoder.

Saves input / reconstructed pairs plus PSNR metrics.
"""
import argparse, csv, os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from main import instantiate_from_config


def load_square(path, box=None, size=256):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if box is not None:
        im = im.crop(box)
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    return im.resize((size, size), Image.BICUBIC)


def to_tensor(im):
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1.0)
    return x.permute(2, 0, 1)[None]


def to_image(x):
    x = ((x[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0)
    return Image.fromarray(x.round().byte().cpu().numpy())


class Stage1Autoencoder(torch.nn.Module):
    """Inference-only VQGAN built from a stage-1 checkpoint (no loss / discriminator)."""

    def __init__(self, ckpt_path, ddconfig, n_embed, embed_dim):
        super().__init__()
        from taming.modules.diffusionmodules.model import Encoder, Decoder
        from taming.modules.vqvae.quantize import VectorQuantizer2
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        sd = {k: v for k, v in sd.items() if not k.startswith("loss.")}
        self.load_state_dict(sd, strict=True)
        print(f"Restored stage-1 autoencoder from {ckpt_path}")

    def encode_real(self, x, quantize=True):
        h = self.quant_conv(self.encoder(x))
        if not quantize:
            return h, 0, [0, 0, 0]
        return self.quantize(h)

    def decode(self, quant):
        return self.decoder(self.post_quant_conv(quant))


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse), mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="logs/landscape2art/configs/test_paired.yaml")
    parser.add_argument("--out", default="outputs/baseline_reconstruction")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--stage1", default=None, help="stage-1 VQGAN checkpoint to use instead of the artwork autoencoder")
    parser.add_argument("images", nargs="+", help="image paths, optionally path:left,top,right,bottom crop box")
    opt = parser.parse_args()

    config = OmegaConf.load(opt.config)
    if opt.stage1:
        p = config.model.params
        model = Stage1Autoencoder(opt.stage1, p.ddconfig, p.n_embed, p.embed_dim).eval()
    else:
        model = instantiate_from_config(config.model).eval()

    rows = []
    for spec in opt.images:
        path, box = spec, None
        if ":" in spec:
            path, b = spec.rsplit(":", 1)
            box = tuple(int(v) for v in b.split(","))
        name = os.path.splitext(os.path.basename(path))[0]
        im = load_square(path, box, opt.size)
        x = to_tensor(im)

        for mode, quantize in (("quantized", True), ("continuous", False)):
            with torch.no_grad():
                h, _, _ = model.encode_real(x, quantize=quantize)
                rec = to_image(model.decode(h))
            for sub, img in (("input", im), ("reconstructed", rec)):
                d = os.path.join(opt.out, mode, sub)
                os.makedirs(d, exist_ok=True)
                img.save(os.path.join(d, name + ".png"))
            p, mse = psnr(im, rec)
            rows.append({"image": name, "mode": mode, "psnr_db": round(p, 2), "mse": round(mse, 2)})
            print(f"{name:24s} {mode:10s} PSNR {p:6.2f} dB")

    with open(os.path.join(opt.out, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mode", "psnr_db", "mse"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
