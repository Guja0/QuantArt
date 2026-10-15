# QuantArt
Official PyTorch implementation of the paper:

[**QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity**](https://arxiv.org/abs/2212.10431)  
[Siyu Huang<sup>*</sup>](https://siyuhuang.github.io) (Harvard), [Jie An<sup>*</sup>](https://www.cs.rochester.edu/u/jan6/) (Rochester), [Donglai Wei](https://donglaiw.github.io/) (BC), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/) (Rochester), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people/hanspeter-pfister) (Harvard)  
CVPR 2023

We devise a new style transfer framework called QuantArt for high visual-fidelity stylization. The core idea is to push latent representation of  generated artwork toward centroids of real artwork distribution with vector quantization. QuantArt achieves decent performance for various image style transfer tasks.

<p align='center'>
 <img alt='Thumbnail' src='imgs/thumb.png'>
</p>

## About this repository
This is a copy of the official QuantArt code with an added experiment: fine-tuning QuantArt's **continuous artwork decoder** on a small custom artwork set (180 photographed artworks, not included in the repo). The original code and history are kept; the additions are:

| File | Purpose |
| ---- | ---- |
| `make_cu_split.py` | Converts the artworks to JPEG and splits them 140 / 20 / 20 (train / val / test), keeping duplicate and near-duplicate images in the same split. The resulting lists are in `datasets/CU_data_split/`. |
| `finetune_artwork_decoder.py` | Trains the continuous artwork encoder and decoder together by default, with L1 + LPIPS + multi-scale edge loss, warmup/cosine scheduling, EMA and TensorBoard. Optional encoder-only and decoder-only modes remain available. No discriminator is used. |
| `compare_decoders.py` | Evaluates the original and the fine-tuned decoder once on the test set (L1 / PSNR / LPIPS), then runs QuantArt style transfer with identical content–style pairs through both decoders and scores it with latent proxies plus a VGG Gram style distance and LPIPS to the content photo. |
| `reconstruct_artwork.py` | Reconstructs images with a single autoencoder (the artwork autoencoder, or a Stage-1 VQGAN via `--stage1`), with and without vector quantization. |
| `main.py` | One change: falls back to CPU when CUDA is unavailable. |

The new scripts use plain PyTorch (see `requirements-train.txt`) and run on Apple Silicon (MPS).

```
python make_cu_split.py
python finetune_artwork_decoder.py --train_part both --edge_weight 0.2 --run joint_ae_60ep
python finetune_artwork_decoder.py --train_part decoder --edge_weight 0 --run l1_lpips_60ep_ema
python compare_decoders.py --run outputs/finetune_decoder/l1_lpips_60ep_ema
tensorboard --logdir outputs/finetune_decoder
```

By default the scripts load the `landscape2art` checkpoint from `landscape2art/` and use its artwork autoencoder with the codebook bypassed; pass `--config` / `--ckpt` to use another checkpoint. On the 20 test artworks, L1-only fine-tuning raised PSNR (21.0 → 23.5 dB) but blurred the images (LPIPS 0.362 → 0.472, lower is better). L1 + LPIPS gave 22.7 dB and LPIPS 0.340 while keeping texture. Those runs used 10 epochs at lr 3e-6 (about 350 steps), fixed 288 px crops and no EMA or GAN; the closest equivalent now is `--train_part decoder --edge_weight 0 --epochs 10 --lr 3e-6 --min_lr_ratio 1 --warmup_steps 0 --scale_max 1.125 --ema 0 --select_metric loss --patience 2` (crops are now drawn from 256–288 px instead of always 288 px).

## Full comparison of original and fine-tuned autoencoders

`compare_all_autoencoders.py` evaluates every artwork with matched deterministic inputs. The primary comparison uses original coco2art and the **final epoch-60 EMA checkpoints** from decoder-only, encoder-only, and joint training. Supplementary center-crop comparisons include the best encoder/joint checkpoints, earlier GAN experiments, and the all-images epoch-87 decoder.

```bash
.venv/bin/python -u compare_all_autoencoders.py --device cuda \
  --out outputs/model_comparison_all_artworks
```

Outputs include an offline `comparison.html` browser, a detailed `RESULTS.md`, model provenance in `summary.json`, per-image and aggregate CSV metrics, and individual PNG reconstructions. Each of the 180 artworks has labeled comparisons and fixed-scale error maps for three views: full composition fitted to 512 pixels; the validation-style 256-pixel center crop; and a 256-pixel detail patch selected by target gradient energy from nine candidate patches at up to 640-pixel short-side resolution. The input selection is identical across models. Train/validation/test metrics are separate.

Interpret the encoder-only model as **updated encoder plus previously fine-tuned frozen decoder**. Comparing it to decoder-only isolates its encoder update; comparing it to original includes changes to both sides. Joint training also adds edge loss, so this is an experiment comparison rather than a controlled architecture ablation. The epoch-87 decoder used all artworks for training, making even its nominal test-subset reconstruction in-sample. Higher PSNR or lower LPIPS alone does not establish better brushwork: inspect target-matching edge/Laplacian errors, sharpness ratio, and the actual detail crops.

Run from the `QuantArt` directory using the existing CUDA environment. The completed comparison refuses to overwrite its output folder; use a different `--out` for a later evaluation. This script performs evaluation only.

## Joint artwork autoencoder fine-tuning (no discriminator)

The default is now `--train_part both`: train `encoder_real`, `quant_conv_real`, `post_quant_conv`, and `decoder` together. No discriminator is created or trained; the GAN losses, flags, and selection gating have been removed from the fine-tuning script. QuantArt's photo encoder, transfer module, and codebooks remain outside this reconstruction experiment.

The loss is L1 + LPIPS + 0.2 × multi-scale target-gradient matching. `--edge_weight` defaults to 0.2; set it to 0 to disable that penalty. Training and validation log unweighted edge error separately. Warmup/cosine scheduling, EMA, TensorBoard reconstruction grids, and single-side encoder/decoder modes remain available. Existing GAN experiment artifacts are preserved as historical results.

Run from `QuantArt`:

```bash
.venv/bin/python -u finetune_artwork_decoder.py \
  --config configs/coco2art.yaml \
  --ckpt ../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt \
  --data ../datasets/CU_data_jpg --split datasets/CU_data_split \
  --train_part both --edge_weight 0.2 \
  --epochs 60 --patience 61 --batch_size 4 --device cuda \
  --run joint_coco2art_edge_60ep
```

The run saves `best_autoencoder.pt` and `last_autoencoder.pt` with **both sides**, along with matching `best_encoder.pt`/`best_decoder.pt` and `last_encoder.pt`/`last_decoder.pt` exports. `last_autoencoder_raw.pt` contains the final non-EMA model. Use the complete autoencoder checkpoint or matching encoder/decoder exports from the same run and epoch.

`compare_decoders.py --run ...` prefers `best_autoencoder.pt` when present. For photo transfer, use `stylize_pair.py --autoencoder .../best_autoencoder.pt` with the original config/checkpoint and your content/style paths. Changing the artwork encoder changes the reference features used by the pretrained transfer module, so assess photo transfer separately from reconstruction. The latent visualization script also recognizes joint checkpoints.

The one-epoch GPU verification run is `outputs/finetune_decoder/joint_autoencoder_verify_20261014/`. The 60-epoch command above is provided for starting a full experiment separately.

## Artwork latent-space comparisons

The baseline and every completed fine-tuning run are compared in `outputs/latent_visualization/latent_explorer.html`. This standalone HTML includes image thumbnails, shared PCA and joint t-SNE maps, model selectors, six baseline-anchored appearance groups, and eight nearest neighbors for each artwork. Choose the right-hand model, click a point, and inspect the nearest-image cards or group gallery. Turn on the thumbnail overlay to see paintings directly on the map.

Two descriptors are available: latent Gram channel correlations (appearance/texture) and channel means/standard deviations. All 180 artworks use identical full-image resize/padding to 256 pixels. Padding is masked when pooling descriptors. Decoder-only models retain the same encoder and therefore have identical latents; encoder epoch 60, best epoch 55, and raw epoch 60 are distinct comparisons. All unique encoders share the fitted projections; neighbors are computed in the original normalized descriptor space, not the 2D map.

These are exploratory appearance groups, not verified oil/acrylic or brushstroke classifications. The six-group choice is a display default. Palette, composition, frames, photographed lighting, museum labels, and duplicate artworks can affect groups. A well-separated-looking t-SNE plot does not establish semantic clustering or improvement from fine-tuning. True medium/brushstroke grouping requires labels to evaluate against.

Run from `QuantArt`:

```bash
.venv/bin/python -m pip install -r requirements-visualization.txt
.venv/bin/python -u visualize_artwork_latents.py \
  --config configs/coco2art.yaml \
  --ckpt ../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt \
  --data ../datasets/CU_data_jpg --device cuda
```

The output folder also contains `baseline_vs_encoder_pca.png`, representative-image contact sheets under `clusters/`, `coordinates_and_neighbors.csv`, `latent_descriptors_and_coordinates.npz`, and `summary.json` with checkpoint hashes, preprocessing, projection variance, group sizes, and nearest-neighbor overlap. The explorer embeds Plotly and thumbnails, so it works offline without external assets.

## Encoder-only experiment with a frozen fine-tuned decoder

The `encoder_coco2art_60ep` experiment trains `encoder_real` and `quant_conv_real` for 60 epochs while keeping the existing 60-epoch artwork decoder and `post_quant_conv` fixed. It uses the 140/20 train/validation split, L1 + LPIPS, EMA, and no early stopping during those 60 epochs. TensorBoard at http://localhost:6006 shows this run's `train/*`, `val/*`, and `val/input_baseline_current` images (rows: input, initial reconstruction, current reconstruction).

From `QuantArt`, the equivalent command is:

```bash
.venv/bin/python -u finetune_artwork_decoder.py \
  --config configs/coco2art.yaml \
  --ckpt ../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt \
  --decoder outputs/finetune_decoder/dev_coco2art_60ep/checkpoints/last_decoder.pt \
  --data ../datasets/CU_data_jpg --split datasets/CU_data_split \
  --train_part encoder --run encoder_coco2art_60ep \
  --epochs 60 --patience 61 --batch_size 4 --device cuda
```

Do not start this command again while the existing background run is active. Its console log is `outputs/finetune_decoder/encoder_coco2art_60ep.log`. The run saves `best_encoder.pt`, a per-epoch `last_encoder.pt`, and the exact `frozen_decoder.pt` needed with those encoder exports. The existing decoder files are not modified. A one-batch GPU check recorded in `outputs/finetune_encoder/verification.json` verified encoder updates, absent decoder gradients, and bit-identical frozen decoder weights.

## Photo-to-art inference with the 60-epoch dev decoder

`stylize_pair.py` runs the complete continuous QuantArt pipeline with your fine-tuned artwork decoder. It accepts JPEG/PNG and HEIC images (`pillow-heif` is included in `requirements-train.txt`). Both inputs are resized and edge-padded to matching square canvases, as required by this branch's cross-attention; padding is removed from the saved photo output. The default canvas is 512×512, preserving the original composition and aspect ratio at reduced resolution.

From `QuantArt`, run:

```bash
.venv/bin/python -u stylize_pair.py \
  --config configs/coco2art.yaml \
  --ckpt ../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt \
  --decoder outputs/finetune_decoder/dev_coco2art_60ep/checkpoints/last_decoder.pt \
  --content test/style_image.HEIC --style test/style_baseline.heic \
  --out outputs/style_transfer/dev_coco2art_60ep_test --size 512 --device cuda
```

The output folder contains `finetuned_quantart.png`, `original_quantart.png`, and `comparison_photo_art_original_finetuned.png` (panels: photo, artwork, original model, fine-tuned model). `run.json` records the checkpoint, input paths, preprocessing, and dimensions. This uses the codebook-free path matching decoder fine-tuning; `last_decoder.pt` contains the EMA decoder saved at epoch 60.

## Encoder-only fine-tuning (decoder frozen)

`finetune_artwork_decoder.py --train_part encoder` freezes `post_quant_conv` + `decoder` and trains `encoder_real` + `quant_conv_real` (checkpoints: `best_encoder.pt`, `last_encoder.pt`). `--epochs 0` saves the untouched weights and baseline metrics without training. `compare_decoders.py` detects encoder checkpoints, and `stylize_pair.py` takes `--encoder` and/or `--decoder`. In style transfer the artwork encoder only encodes the style reference; the photo goes through the photo `encoder`, which stays untouched.

`run_encoder_only.sh` runs the baseline, training, comparison and 512 px stylization in one go:

```bash
CKPT=../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt bash run_encoder_only.sh
# smoke test first:
CKPT=... RUN=enc_smoke EXTRA="--epochs 1 --max_steps 2" bash run_encoder_only.sh
```

## 🔴Update (07/20/2025)
I have fixed all the links to the pretrained models, but I cannot guarantee their weights are 100% correct. Feel free to report any issues!

## Dependencies
* python=3.8.5
* pytorch=1.7.0
* pytorch-lightning=1.0.8
* cuda=10.2
 
We recommend to use `conda` to create a new environment with all dependencies installed.
```
conda env create -f environment.yaml
conda activate quantart
```

## Quick Example of Landscape Style Transfer
Download pre-trained [landscape2art model](https://drive.google.com/drive/folders/1uTNhiLRzERKEYHGTlAEWhTOUvJDD3ibt?usp=drive_link) and put it under `logs/`. Run
```
bash test.sh
```
The stylized landscape images (from `imgs/`) will be saved in `logs/`.

## Datasets and Pre-trained Models

**Stage-1:** The datasets and pre-trained models for codebook pretraining are as follows:

| Dataset | Pre-trained Model |
| ---- | ---- |
| [MS_COCO](https://cocodataset.org/#download) | [vqgan_imagenet_f16_1024.ckpt](https://drive.google.com/file/d/1TQRfO1InznRME2RlyKwzQRIb0xcalOiP/view?usp=drive_link) |
| [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | [vqgan_wikiart_f16_1024.ckpt](https://drive.google.com/file/d/1YXVNlQhKHEcFZFGg_2127OP09qnviCc-/view?usp=drive_link) |
| [LandscapesHQ](https://github.com/universome/alis) | [vqgan_landscape_f16_1024.ckpt](https://drive.google.com/file/d/1yeWjTewg3sIU1LPBmSz-k3XMGYmqd9hI/view?usp=drive_link) |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [vqgan_faceshq_f16_1024.ckpt](https://drive.google.com/file/d/1yn6ZISlsJcQqwz3F44Mu1t29IGfGV6eg/view?usp=drive_link) |
| [Metfaces](https://github.com/NVlabs/metfaces-dataset) | [vqgan_metfaces_f16_1024.ckpt](https://drive.google.com/file/d/1EEIjsFc1OdyzVUdX2rxw-IaotS3Tq2l7/view?usp=drive_link) |

**Stage-2:** The datasets and pre-trained models for style transfer experiments are as follows:

| Task | Pre-trained Model | Content | Style |
| ---- | ---- | ---- | ---- |
| photo->artwork| [coco2art](https://drive.google.com/drive/folders/1s_3X7j8xIMIh8DjIg3-HPVp_zQSr6uxr?usp=drive_link) | [MS_COCO](https://cocodataset.org/#download) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| landscape->artwork | [landscape2art](https://drive.google.com/drive/folders/1uTNhiLRzERKEYHGTlAEWhTOUvJDD3ibt?usp=drive_link)| [LandscapesHQ](https://github.com/universome/alis) |[WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| landscape->artwork (non-VQ) | [landscape2art_continuous](https://drive.google.com/drive/folders/1zRLlaLWtucSg4hWbvGmCaj6OtAdWA_Xp?usp=drive_link)| [LandscapesHQ](https://github.com/universome/alis) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |  
| face->artwork | [face2art](https://drive.google.com/drive/folders/11IQwT4XMrqKxozPnDf-pOSP3mHN1liKe?usp=drive_link) | [FFHQ](https://github.com/NVlabs/ffhq-dataset) | [Metfaces](https://github.com/NVlabs/metfaces-dataset) |
| artwork->artwork | [art2art](https://drive.google.com/drive/folders/1bg7iGaodQ6R7JGlNTTFeV8QniF5OA0n_?usp=drive_link) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) | [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) |
| photo->photo | [coco2coco](https://drive.google.com/drive/folders/1D1vfKsKVzB115_QmxWCqAwHQpGETQAlp?usp=drive_link) | [MS_COCO](https://cocodataset.org/#download) | [MS_COCO](https://cocodataset.org/#download) |
| landscape->landscape | [landscape2landscape](https://drive.google.com/drive/folders/1JTfYu8Zr5oOYPA1RCbczcp66vBqT9rxV?usp=drive_link) | [LandscapesHQ](https://github.com/universome/alis) | [LandscapesHQ](https://github.com/universome/alis) |

## Testing
Follow **Datasets and Pre-trained Models** to download more datasets and pretrained models. For instance for landscape-to-artwork style transfer model, the folder structure should be
```
QuantArt
├── configs
├── datasets
│   ├── lhq_1024_jpg
│   │   ├── lhq_1024_jpg
│   │   │   ├── 0000000.jpg
│   │   │   ├── 0000001.jpg
│   │   │   ├── 0000002.jpg
│   │   │   ├── ...
│   ├── painter-by-numbers
│   │   ├── train
│   │   │   ├── 100001.jpg
│   │   │   ├── 100002.jpg
│   │   │   ├── 100003.jpg
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── 0.jpg
│   │   │   ├── 100000.jpg
│   │   │   ├── 100004.jpg
│   │   │   ├── ...
├── logs
│   ├── landscape2art
│   │   ├── checkpoints
│   │   ├── configs
├── taming
├── environment.yaml
├── main.py
├── train.sh
└── test.sh
```

Run the following command to test the pre-trained model on the testing dataset:
```
python -u main.py --base logs/landscape2art/configs/test.yaml -n landscape2art -t False --gpus 0,
```

* `--base`: path for the config file.
* `-n`: result folder under `logs/`.
* `-t`: is training.
* `--gpus`: GPUs used.


## Training
**Stage-1:** Prepare [WikiArt](https://www.kaggle.com/competitions/painter-by-numbers/data) dataset as above. Download file lists [painter-by-numbers-train.txt](https://drive.google.com/file/d/1tNoLcX9q-Yty5PcpMNod1WXY208ZctHz/view?usp=drive_link) and [painter-by-numbers-test.txt](https://drive.google.com/file/d/19oi3x5bwiniZ-7FYyG4H26pve0kr9vpM/view?usp=drive_link), put them under `datasets/`. Run the following command to train a Stage-1 model (i.e., an autoencoder and a codebook). Four GPUs are recommended but not necessary.
```
python -u main.py --base configs/vqgan_wikiart.yaml -t True --gpus 0,1,2,3
```

Two separate Stage-1 models are required for content and style datasets, respectively.

**Stage-2:** Run `bash train.sh` or the following command to train a photo-to-artwork model
```
python -u main.py --base configs/coco2art.yaml -t True --gpus 0,
```

* `--base`: path for the config file.
* `-n`: result folder under `logs/`.
* `-t`: is training.
* `--gpus`: GPUs used.
* `--resume_from_checkpoint`: resume training from a checkpoint.

More training configs of Stage-2 models can be found in `configs/`.

## Custom Dataset
**Unpaired data:**
To test unpaired data, follow comments in `configs/custom_unpaired.yaml` to specify model checkpoints and data paths. Then run
```
python -u main.py --base configs/custom_unpaired.yaml -n custom_unpaired -t False --gpus 0,
```
**Paired data:**
To test paired data, the corresponding content and style images (in two folders) should have the same file names. Follow comments in `configs/custom_paired.yaml` to specify model checkpoints and data paths, then run
```
python -u main.py --base configs/custom_paired.yaml -n custom_paired -t False --gpus 0,
```

## Citation
```
@inproceedings{huang2023quantart,
    title={QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity},
    author={Siyu Huang and Jie An and Donglai Wei and Jiebo Luo and Hanspeter Pfister},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    month={June},
    year={2023}
}
```

## Acknowledgement
This repository is heavily built upon the amazing [VQGAN](https://github.com/CompVis/taming-transformers).

## Contact
[Siyu Huang](http://siyuhuang.github.io/) (huangsiyutc@gmail.com).
