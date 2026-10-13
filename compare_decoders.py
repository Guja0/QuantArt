"""Compare the original and the fine-tuned continuous artwork decoder.

1. Test reconstruction: the 20 held-out test artworks through the frozen artwork
   encoder and each decoder (L1 / PSNR / LPIPS, saved images).
2. Style transfer inside QuantArt (continuous model): identical content-style pairs,
   photo encoder -> StyleTransferModule(ref = artwork encoder) -> decoder, once with
   each decoder. Saved as grids; two proxy metrics computed with the frozen encoders:
     style_gram   Gram-matrix distance between output and reference in artwork-encoder
                  feature space (lower = closer to the reference style)
     content_l1   L1 between content and output in photo-encoder latent space
                  (lower = more content preserved)
"""
import argparse, csv, json, os

import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf
from PIL import Image

from finetune_artwork_decoder import (ArtworkDataset, ContinuousArtworkAE, load_lpips, load_state_dict,
                                      pick_device, psnr, resize_shorter, to_image, to_tensor)
from taming.modules.diffusionmodules.model import Decoder, Encoder, StyleTransferModule

DEFAULT_CONTENT = ["imgs/content/1.jpg", "imgs/content/2.jpg", "imgs/content/3.jpg", "imgs/content/4.jpg",
                   "imgs_reconstruction_photo/style_image.jpg", "imgs_reconstruction_photo/style_image_2.jpg",
                   "imgs_reconstruction_photo/olive_trees.jpg"]


class ContinuousQuantArt(torch.nn.Module):
    """Inference-only QuantArt continuous model (taming.models.vqgan_ref_continuous)."""

    def __init__(self, ddconfig, embed_dim, use_residual=True):
        super().__init__()
        z = ddconfig["z_channels"]
        self.encoder, self.encoder_real, self.decoder = Encoder(**ddconfig), Encoder(**ddconfig), Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(z, embed_dim, 1)
        self.quant_conv_real = torch.nn.Conv2d(z, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z, 1)
        self.model_x2y = StyleTransferModule(embed_dim, block_num=6, residual=use_residual)

    @classmethod
    def from_checkpoint(cls, config_path, ckpt_path):
        p = OmegaConf.load(config_path).model.params
        model = cls(p.ddconfig, p.embed_dim, p.get("use_residual", True))
        sd = load_state_dict(ckpt_path)
        keep = tuple(n + "." for n, _ in model.named_children())
        model.load_state_dict({k: v for k, v in sd.items() if k.startswith(keep)}, strict=True)
        return model

    @torch.no_grad()
    def transfer(self, x, ref):
        h_x = self.quant_conv(self.encoder(x))
        h_ref = self.quant_conv_real(self.encoder_real(ref))
        return self.decoder(self.post_quant_conv(self.model_x2y(h_x, h_ref)))

    @torch.no_grad()
    def gram(self, x):
        h = self.quant_conv_real(self.encoder_real(x)).flatten(2)
        return h @ h.transpose(1, 2) / h.shape[-1]

    @torch.no_grad()
    def content_latent(self, x):
        return self.quant_conv(self.encoder(x))


def load_fixed(path, size=256):
    im = resize_shorter(Image.open(path).convert("RGB"), size)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return to_tensor(im.crop((left, top, left + size, top + size)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="landscape2art/configs/test.yaml")
    parser.add_argument("--ckpt", default="landscape2art/checkpoints/last.ckpt")
    parser.add_argument("--run", required=True, help="fine-tuning run dir (contains checkpoints/best_decoder.pt)")
    parser.add_argument("--decoder", default=None, help="adapted decoder weights (default: <run>/checkpoints/best_decoder.pt)")
    parser.add_argument("--data", default="datasets/CU_data_jpg")
    parser.add_argument("--split", default="datasets/CU_data_split")
    parser.add_argument("--content", nargs="+", default=DEFAULT_CONTENT)
    parser.add_argument("--n_styles", type=int, default=5, help="first N test artworks used as style references")
    parser.add_argument("--device", default="auto")
    opt = parser.parse_args()

    device = pick_device(opt.device)
    out = os.path.join(opt.run, "comparison")
    adapted_sd = torch.load(opt.decoder or os.path.join(opt.run, "checkpoints", "best_decoder.pt"), map_location="cpu",
                            weights_only=True)

    # ---- 1. test reconstruction --------------------------------------------------
    test_set = ArtworkDataset(opt.data, os.path.join(opt.split, "test.txt"))
    ae = ContinuousArtworkAE.from_checkpoint(opt.config, opt.ckpt).to(device).eval()
    original_sd = {k: v.clone() for k, v in ae.decoder_state_dict().items()}
    x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    names = [os.path.splitext(n)[0] for n in test_set.names]
    recs = {}
    for tag, sd in (("original", original_sd), ("adapted", adapted_sd)):
        ae.load_state_dict(sd, strict=False)
        with torch.no_grad():
            recs[tag] = torch.cat([ae.decode(ae.encode(b.to(device))).cpu() for b in x.split(4)])
    lpips = load_lpips(device)
    rows = []
    for i, n in enumerate(names):
        row = {"image": n}
        for tag in ("original", "adapted"):
            row[f"{tag}_l1"] = F.l1_loss(recs[tag][i], x[i]).item()
            row[f"{tag}_psnr"] = psnr(recs[tag][i:i + 1], x[i:i + 1]).item()
            with torch.no_grad():
                row[f"{tag}_lpips"] = lpips(recs[tag][i:i + 1].to(device), x[i:i + 1].to(device)).item()
        rows.append(row)
    for sub, batch in (("input", x), ("original", recs["original"]), ("adapted", recs["adapted"])):
        os.makedirs(os.path.join(out, "test_reconstruction", sub), exist_ok=True)
        for n, img in zip(names, batch):
            to_image(img).save(os.path.join(out, "test_reconstruction", sub, n + ".png"))
    torchvision.utils.save_image(
        torchvision.utils.make_grid((torch.cat([x, recs["original"], recs["adapted"]]).clamp(-1, 1) + 1) / 2,
                                    nrow=len(x), padding=2),
        os.path.join(out, "test_reconstruction", "grid_input_original_adapted.png"))
    with open(os.path.join(out, "test_reconstruction", "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"test": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "image"}}
    print("test reconstruction:", json.dumps(summary["test"], indent=1))
    del ae

    # ---- 2. style transfer -------------------------------------------------------
    qa = ContinuousQuantArt.from_checkpoint(opt.config, opt.ckpt).to(device).eval()
    styles = x[:opt.n_styles]
    style_names = names[:opt.n_styles]
    contents = torch.stack([load_fixed(p) for p in opt.content])
    content_names = [os.path.splitext(os.path.basename(p))[0] for p in opt.content]
    st_rows, outputs = [], {}
    for tag, sd in (("original", original_sd), ("adapted", adapted_sd)):
        qa.decoder.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
        qa.post_quant_conv.load_state_dict({k[len("post_quant_conv."):]: v for k, v in sd.items()
                                            if k.startswith("post_quant_conv.")})
        outputs[tag] = torch.stack([torch.stack([qa.transfer(c[None].to(device), s[None].to(device))[0].cpu()
                                                 for s in styles]) for c in contents])  # [C, S, 3, H, W]
    for ci, cn in enumerate(content_names):
        for si, sn in enumerate(style_names):
            ref, content = styles[si:si + 1].to(device), contents[ci:ci + 1].to(device)
            row = {"content": cn, "style": sn}
            for tag in ("original", "adapted"):
                y = outputs[tag][ci, si:si + 1].to(device)
                row[f"{tag}_style_gram"] = (qa.gram(y) - qa.gram(ref)).pow(2).mean().sqrt().item()
                row[f"{tag}_content_l1"] = F.l1_loss(qa.content_latent(y), qa.content_latent(content)).item()
            st_rows.append(row)
    st_dir = os.path.join(out, "style_transfer")
    os.makedirs(st_dir, exist_ok=True)
    for si, sn in enumerate(style_names):
        # one grid per style: rows = content | original output | adapted output, columns = contents
        torchvision.utils.save_image(
            torchvision.utils.make_grid((torch.cat([contents, outputs["original"][:, si], outputs["adapted"][:, si]])
                                         .clamp(-1, 1) + 1) / 2, nrow=len(contents), padding=2),
            os.path.join(st_dir, f"style_{sn}_rows_content_original_adapted.png"))
        to_image(styles[si]).save(os.path.join(st_dir, f"style_{sn}_reference.png"))
    with open(os.path.join(st_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(st_rows[0]))
        w.writeheader()
        w.writerows(st_rows)
    summary["style_transfer"] = {k: sum(r[k] for r in st_rows) / len(st_rows) for k in st_rows[0]
                                 if k not in ("content", "style")}
    print("style transfer:", json.dumps(summary["style_transfer"], indent=1))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
