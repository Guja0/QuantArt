# Photo-to-art comparison

Open `index.html` for all paired comparisons. Each grid shows photo, selected artwork and original output in the first row, then decoder-only, encoder-only (with its frozen tuned decoder) and joint output in the second row. Model folders contain lossless outputs with padding removed; comparisons retain a neutral border for alignment. No retraining was performed.

`run.json` records the random seed, selected artwork per cluster, original filenames, checkpoint paths and hashes. All three experiments use final epoch-60 EMA weights. The original photo encoder and transfer module remain fixed. Appearance clusters are not verified labels for oil paint or brushstroke types. These are qualitative transfer results, with no ground-truth stylized target.
