"""Compare epoch-60 experiments on photos and seeded samples of baseline artwork clusters."""
import argparse
import csv
import hashlib
import html
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from compare_decoders import ContinuousQuantArt
from finetune_artwork_decoder import DECODER_PREFIXES, ENCODER_PREFIXES, to_image, to_tensor
from stylize_pair import fit_square, load_image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def panel(image, label, size):
    tile = Image.new('RGB', (size, size + 30), 'white')
    tile.paste(image, (0, 30))
    ImageDraw.Draw(tile).text((10, 9), label, fill='black')
    return tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--photos', required=True)
    parser.add_argument('--artworks', required=True)
    parser.add_argument('--clusters', required=True, help='coordinates_and_neighbors.csv')
    parser.add_argument('--runs', default='outputs/finetune_decoder')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--descriptor', choices=['gram', 'statistics'], default='gram')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.size < 32 or args.size % 16:
        parser.error('--size must be at least 32 and divisible by 16')
    out = Path(args.out)
    if (out / 'run.json').exists():
        parser.error('output run already exists; choose a new --out directory')
    groups = {}
    with open(args.clusters, newline='') as f:
        for row in csv.DictReader(f):
            if row['model'] == 'Baseline coco2art' and row['descriptor'] == args.descriptor:
                groups.setdefault(int(row['cluster']), []).append(row['image'])
    if not groups:
        raise ValueError('No baseline clusters found')
    rng = random.Random(args.seed)
    selected = [{'cluster': c, 'image': rng.choice(sorted(set(names))),
                 'cluster_size': len(set(names))} for c, names in sorted(groups.items())]
    photos = sorted(p for p in Path(args.photos).iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif'}
                    and not p.name.startswith('._'))
    if not photos:
        raise ValueError('No photos found')
    runs = Path(args.runs)
    cases = {
        'original': ('Original coco2art', []),
        'decoder_only': ('Decoder only / epoch 60', [runs / 'dev_coco2art_60ep/checkpoints/last_decoder.pt']),
        'encoder_only': ('Encoder only + frozen tuned decoder / epoch 60', [
            runs / 'encoder_coco2art_60ep/checkpoints/frozen_decoder.pt',
            runs / 'encoder_coco2art_60ep/checkpoints/last_encoder.pt']),
        'joint': ('Joint encoder + decoder / epoch 60', [
            runs / 'joint_coco2art_edge_60ep/checkpoints/last_autoencoder.pt']),
    }
    # Decode every supplied photo and chosen reference before GPU inference.
    prepared_photos = [(p, *fit_square(load_image(p), args.size)) for p in photos]
    prepared_styles = [(s, fit_square(load_image(Path(args.artworks) / s['image']), args.size)[0])
                       for s in selected]
    metadata = dict(settings=vars(args), status='running', selected_artworks=selected,
                    photos=[str(p) for p in photos], models={}, completed_outputs=0,
                    notes=[
                        'Identical photo/artwork pairs and preprocessing for every model.',
                        'Continuous photo encoder -> pretrained transfer module -> decoder; no codebooks.',
                        'Encoder fine-tuning changes the artwork-reference encoder, not the photo encoder.',
                        'Encoder-only uses its saved frozen, already-tuned decoder.',
                        'Joint used edge loss; this is not a controlled single-variable ablation.',
                        'Clusters are exploratory appearance groups, not verified painting-medium labels.',
                        'Inputs fit within a 512px canvas by default with edge padding; output padding removed.',
                    ])
    for tag, (label, weights) in cases.items():
        metadata['models'][tag] = dict(label=label, weights=[dict(path=str(p), sha256=digest(p)) for p in weights])
    metadata['pretrained_sha256'] = digest(args.ckpt)
    metadata['cluster_csv_sha256'] = digest(args.clusters)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ('photos', 'artworks', 'comparisons', *cases):
        (out / sub).mkdir(exist_ok=True)
    for p, image, _ in prepared_photos:
        image.save(out / 'photos' / (p.stem + '.jpg'), quality=95)
    for s, image in prepared_styles:
        image.save(out / 'artworks' / f"cluster_{s['cluster']}_{Path(s['image']).stem}.jpg", quality=95)
    def save_metadata():
        (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    save_metadata()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with GPU access')
    torch.manual_seed(args.seed)
    qa = ContinuousQuantArt.from_checkpoint(args.config, args.ckpt).eval().requires_grad_(False).to(device)
    prefixes = ENCODER_PREFIXES + DECODER_PREFIXES
    original = {k: v.detach().cpu().clone() for k, v in qa.state_dict().items() if k.startswith(prefixes)}
    decoder = torch.load(cases['decoder_only'][1][0], map_location='cpu', weights_only=True)
    frozen = torch.load(cases['encoder_only'][1][0], map_location='cpu', weights_only=True)
    if decoder.keys() != frozen.keys() or any(not torch.equal(decoder[k], frozen[k]) for k in decoder):
        raise ValueError('Encoder-only frozen decoder differs from decoder-only checkpoint')
    del decoder, frozen
    metadata['encoder_frozen_decoder_matches_decoder_only'] = True
    with torch.inference_mode():
        for tag, (_, paths) in cases.items():
            qa.load_state_dict(original, strict=False)
            loaded = set()
            for path in paths:
                state = torch.load(path, map_location='cpu', weights_only=True)
                expected_prefixes = prefixes if path.name.endswith('autoencoder.pt') else (
                    ENCODER_PREFIXES if 'encoder' in path.name else DECODER_PREFIXES)
                expected = {k for k in original if k.startswith(expected_prefixes)}
                if set(state) != expected or loaded.intersection(state):
                    raise ValueError(f'Invalid or overlapping checkpoint keys: {path}')
                qa.load_state_dict(state, strict=False)
                loaded.update(state)
            for p, photo, box in prepared_photos:
                x = to_tensor(photo).unsqueeze(0).to(device)
                for s, artwork in prepared_styles:
                    ref = to_tensor(artwork).unsqueeze(0).to(device)
                    y = qa.transfer(x, ref)[0].cpu()
                    if not torch.isfinite(y).all():
                        raise RuntimeError(f'Nonfinite result: {tag} {p.name} {s}')
                    to_image(y).crop(box).save(out / tag / f"{p.stem}__cluster_{s['cluster']}.png")
                    metadata['completed_outputs'] += 1
                save_metadata()
                print(f"{tag}: {p.name}; {metadata['completed_outputs']}/{len(cases)*len(photos)*len(selected)} outputs", flush=True)
    cards = []
    for p, photo, box in prepared_photos:
        for s, artwork in prepared_styles:
            grid = Image.new('RGB', (args.size * 3, (args.size + 30) * 2), 'white')
            tiles = [panel(photo, f'Photo: {p.name}', args.size),
                     panel(artwork, f"Cluster {s['cluster']}: {s['image']}", args.size)]
            for tag, (label, _) in cases.items():
                with Image.open(out / tag / f"{p.stem}__cluster_{s['cluster']}.png") as image:
                    square = Image.new('RGB', (args.size, args.size), '#dddddd')
                    square.paste(image, box[:2])
                    tiles.append(panel(square, label, args.size))
            for i, tile in enumerate(tiles):
                grid.paste(tile, ((i % 3) * args.size, (i // 3) * (args.size + 30)))
            name = f"{p.stem}__cluster_{s['cluster']}.jpg"
            grid.save(out / 'comparisons' / name, quality=94)
            cards.append(f'<section data-photo="{html.escape(p.stem)}" data-cluster="{s["cluster"]}"><h2>{html.escape(p.name)} / Cluster {s["cluster"]} / {html.escape(s["image"])}</h2><a href="comparisons/{name}"><img loading="lazy" src="comparisons/{name}"></a></section>')
    photo_options = ''.join(f'<option>{html.escape(p.stem)}</option>' for p in photos)
    cluster_options = ''.join(f'<option>{s["cluster"]}</option>' for s in selected)
    (out / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Amberg photo-to-art comparisons</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%;height:auto}section{background:white;padding:12px;margin:24px 0}select{margin:8px;padding:8px}nav{position:sticky;top:0;background:#eee;padding:10px}</style>
<h1>Amberg: original vs epoch-60 fine-tuning</h1><p>Rows: photo, artwork, original / decoder-only, encoder-only with frozen tuned decoder, joint. Same reference per cluster across all photos and models. Click a comparison to inspect full resolution. Individual lossless PNG outputs are saved in each model folder.</p>
<nav>Photo <select id="photo"><option value="">All</option>''' + photo_options + '''</select>Cluster <select id="cluster"><option value="">All</option>''' + cluster_options + '''</select></nav>''' + ''.join(cards) + '''<script>function filter(){document.querySelectorAll('section').forEach(s=>s.hidden=(photo.value&&s.dataset.photo!==photo.value)||(cluster.value&&s.dataset.cluster!==cluster.value))}photo.onchange=cluster.onchange=filter;</script>''')
    metadata['status'] = 'complete'
    metadata['pairs'] = len(photos) * len(selected)
    save_metadata()
    (out / 'README.md').write_text('''# Photo-to-art comparison

Open `index.html` for all paired comparisons. Each grid shows photo, selected artwork and original output in the first row, then decoder-only, encoder-only (with its frozen tuned decoder) and joint output in the second row. Model folders contain lossless outputs with padding removed; comparisons retain a neutral border for alignment. No retraining was performed.

`run.json` records the random seed, selected artwork per cluster, original filenames, checkpoint paths and hashes. All three experiments use final epoch-60 EMA weights. The original photo encoder and transfer module remain fixed. Appearance clusters are not verified labels for oil paint or brushstroke types. These are qualitative transfer results, with no ground-truth stylized target.
''')
    print(f"Completed. Open {out / 'index.html'}", flush=True)


if __name__ == '__main__':
    main()
