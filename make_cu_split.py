"""Convert CU_data artworks to JPEG and split them 140 / 20 / 20 with duplicates kept together.

Duplicate groups are found with a perceptual hash and then fixed by hand in
DUPLICATE_GROUPS after visual review, so the split is reproducible.
"""
import argparse, glob, json, os, random, subprocess

from PIL import Image, ImageOps

# Reviewed visually: exact re-shots of the same painting, and colour variants of one composition.
DUPLICATE_GROUPS = [
    ["IMG_0480", "IMG_0481"],
    ["IMG_0584", "IMG_0585 2"],
    ["IMG_0668", "IMG_0735"],
    ["IMG_0589", "IMG_0590"],
    ["IMG_0591", "IMG_0592"],
    ["IMG_0728", "IMG_0729"],
]
# Not artworks (a wall label, a wall of small frames): kept, but only in training.
TRAIN_ONLY = ["IMG_0707", "IMG_0709"]


def convert(src_dir, dst_dir, max_side=1024):
    os.makedirs(dst_dir, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(src_dir, "*"))):
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(dst_dir, name + ".jpg")
        if os.path.exists(dst):
            continue
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "95",
                        "-Z", str(max_side), src, "--out", dst], check=True, capture_output=True)
        im = Image.open(dst)
        if im.getexif().get(274, 1) != 1:
            ImageOps.exif_transpose(im).convert("RGB").save(dst, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="CU_data")
    parser.add_argument("--images", default="datasets/CU_data_jpg")
    parser.add_argument("--out", default="datasets/CU_data_split")
    parser.add_argument("--n_val", type=int, default=20)
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    opt = parser.parse_args()

    convert(opt.src, opt.images)
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(opt.images, "*.jpg")))

    grouped = {n for g in DUPLICATE_GROUPS for n in g}
    groups = [list(g) for g in DUPLICATE_GROUPS] + [[n] for n in names if n not in grouped]
    forced_train = [g for g in groups if any(n in TRAIN_ONLY for n in g)]
    free = [g for g in groups if g not in forced_train]
    random.Random(opt.seed).shuffle(free)

    # Fill val and test with whole groups until each reaches its exact size.
    split = {"val": [], "test": []}
    train = [n for g in forced_train for n in g]
    for g in free:
        for key, size in (("test", opt.n_test), ("val", opt.n_val)):
            if len(split[key]) + len(g) <= size:
                split[key] += g
                break
        else:
            train += g
    split["train"] = train

    os.makedirs(opt.out, exist_ok=True)
    for key in ("train", "val", "test"):
        with open(os.path.join(opt.out, key + ".txt"), "w") as f:
            f.write("\n".join(sorted(n + ".jpg" for n in split[key])) + "\n")
        print(f"{key:5s} {len(split[key])}")
    with open(os.path.join(opt.out, "duplicate_groups.json"), "w") as f:
        json.dump({"duplicate_groups": DUPLICATE_GROUPS, "train_only": TRAIN_ONLY, "seed": opt.seed}, f, indent=2)


if __name__ == "__main__":
    main()
