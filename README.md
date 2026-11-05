# Calibrated Uncertainty Quantification on TDC ADMET

Code for the experiments in
[*Calibrated Uncertainty Quantification on TDC ADMET: What Can Be Resolved Across
22 Datasets?*](https://arxiv.org/abs/2611.09373)

## Data

Data and results are available from the
[DTU Data collection](https://doi.org/10.11583/DTU.c.8827534). The analysis-ready
dataset contains the predictions and result tables used in the paper; the other two
contain pipeline and ensemble-member prediction trajectories.

You can use the data if you for instance want to try other calibration methods or
metrics, compare methods across chemical partitions, or examine different
model-selection and ensemble choices.

## Installation

The code was tested with Python 3.12.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,neural-cpu]"
```

The `neural-cpu` extra installs CPU versions of JAX and Optax. For GPU use,
install JAX for the target platform separately.

## Usage

Prepare the source data and features:

```bash
python -m admet_uq.data --task all
python -m admet_uq.featurize --task all
```

Run one outer fold:

```bash
python scripts/run_cell.py --task dili --head lgbm --outer 0 \
  --splits-dir conf/splits_partition_10x5/s1
python scripts/run_cell.py --task dili --head shared --outer 0 \
  --splits-dir conf/splits_partition_10x5/s1
```

Run the tests:

```bash
python -m pytest
```

## Citation and licence

Citation metadata are provided in `CITATION.cff`. The code is released under the
MIT licence. The deposited data are released under CC BY 4.0; included source
tables remain under their original terms.
