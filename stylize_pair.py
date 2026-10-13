"""Run continuous QuantArt photo-to-art transfer with a fine-tuned decoder.

HEIC inputs require pillow-heif. Both inputs are resized and edge-padded to the
same square for QuantArt's cross-attention; output padding is removed afterward.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from compare_decoders import ContinuousQuantArt
from finetune_artwork_decoder import DECODER_PREFIXES, pick_device, to_image, to_tensor


def load_image(path):
    if Path(path).suffix.lower() in {".heic", ".heif"}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError("Install pillow-heif to read HEIC inputs") from exc
        register_heif_opener()
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def fit_square(im, size):
    fitted = ImageOps.contain(im, (size, size), Image.Resampling.LANCZOS)
    w, h = fitted.size
    left, top = (size - w) // 2, (size - h) // 2
    pixels = np.pad(np.asarray(fitted), ((top, size - h - top), (left, size - w - left), (0, 0)), mode="edge")
    return Image.fromarray(pixels), (left, top, left + w, top + h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=512, help="Square inference canvas, multiple of 16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.size < 32 or args.size % 16:
        parser.error("--size must be at least 32 and a multiple of 16")
    device = pick_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with GPU access")
    photo, artwork = load_image(args.content), load_image(args.style)
    content_square, content_box = fit_square(photo, args.size)
    style_square, _ = fit_square(artwork, args.size)
    x = to_tensor(content_square).unsqueeze(0).to(device)
    ref = to_tensor(style_square).unsqueeze(0).to(device)
    qa = ContinuousQuantArt.from_checkpoint(args.config, args.ckpt).to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        original = qa.transfer(x, ref).cpu()[0]
    state = torch.load(args.decoder, map_location="cpu", weights_only=True)
    expected = {k for k in qa.state_dict() if k.startswith(DECODER_PREFIXES)}
    if set(state) != expected:
        raise ValueError("Decoder checkpoint must contain exactly decoder and post_quant_conv weights")
    qa.decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}, strict=True)
    qa.post_quant_conv.load_state_dict({k[len("post_quant_conv."):]: v for k, v in state.items() if k.startswith("post_quant_conv.")}, strict=True)
    with torch.inference_mode():
        tuned = qa.transfer(x, ref).cpu()[0]
    if not torch.isfinite(original).all() or not torch.isfinite(tuned).all():
        raise RuntimeError("Nonfinite photo-transfer output")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    photo.save(out / "content_photo.png")
    artwork.save(out / "artwork_reference.png")
    original_image, tuned_image = to_image(original), to_image(tuned)
    original_image.crop(content_box).save(out / "original_quantart.png")
    tuned_image.crop(content_box).save(out / "finetuned_quantart.png")
    content_square.save(out / "processed_content.png")
    style_square.save(out / "processed_style.png")
    comparison = Image.new("RGB", (args.size * 4, args.size))
    for i, im in enumerate([content_square, style_square, original_image, tuned_image]):
        comparison.paste(im, (args.size * i, 0))
    comparison.save(out / "comparison_photo_art_original_finetuned.png")
    metadata = vars(args).copy()
    metadata.update(mode="continuous", padding="edge; removed from outputs", content_crop=content_box,
                    content_source_size=photo.size, artwork_source_size=artwork.size,
                    output_size=tuned_image.crop(content_box).size)
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved photo-to-art result: {out / 'finetuned_quantart.png'}")
    print(f"Output size: {metadata['output_size']}; device: {device}")


if __name__ == "__main__":
    main()
