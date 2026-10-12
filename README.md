# MIT ADSP — SVHN Digit Recognition (ANN vs CNN)

Low-code lab for the MIT Applied Data Science Program deep learning elective.
Trains two Artificial Neural Networks (ANN-1, ANN-2) and two Convolutional
Neural Networks (CNN-1, CNN-2) on the SVHN single-digit grayscale dataset, and
produces the facts and figures needed for a separate business report (not
included in this repo).

## Repo layout

```
notebooks/        SVHN_Digit_Recognition.ipynb — the lab notebook
data/              SVHN_single_grey1.h5 (not committed; see .gitignore)
report_figs/       PNG figures exported from the notebook
results/           results_pack.md (facts/tables) and notebook.html (executed export)
requirements.txt   pinned Python dependencies
```

## Environment

- Python 3.12 virtual environment at `.venv/` (TensorFlow + tensorflow-metal
  require Python <= 3.12 on Apple Silicon as of TensorFlow 2.19.1 /
  tensorflow-metal 1.2.0).
- Hardware: Apple Silicon GPU via the Metal `PluggableDevice`. The notebook
  prints the TensorFlow-visible GPU device at the top of the run.

Setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place `SVHN_single_grey1.h5` at `data/SVHN_single_grey1.h5`. The dataset is
never committed to git (`.gitignore` excludes `/data/` and `*.h5`).

The file ships three pre-split arrays: `X_train`/`y_train` (42,000 images),
`X_val`/`y_val` (60,000 images), and `X_test`/`y_test` (18,000 images). Only
the train and test arrays are used; validation accuracy for every model comes
from a `validation_split=0.2` carve-out of the training data at fit time, not
the shipped validation split.

## Running the notebook

The notebook is organized top-to-bottom: setup/seeds, data load, data
overview, ANN preprocessing and models (ANN-1, ANN-2), CNN preprocessing and
models (CNN-1, CNN-2), best-model diagnostics, comparison table, and a final
results-pack writer. "Observations" headings are left blank for the report
author.

## Results

See [`results/results_pack.md`](results/results_pack.md) for all facts,
metrics, and figure references, and [`results/notebook.html`](results/notebook.html)
for the fully executed notebook export.
