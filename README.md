# Probabilistic Circuit Networks Code Repository

This repository contains the code for the ProbML 2026 Workshop Paper:
**"Probabilistic Circuit Networks"**.
You can use the Dockerfile (`.docker/Dockerfile`) to build and run the container.

The paper makes use of the ["simple-einet"](https://github.com/braun-steven/simple-einet/tree/main) repository. All files required for running these experiments are included in this repository already.

This README explains how to reproduce results for the 4 experiments included in the paper:

1. BN benchmark: PCN vs learned BN on standard BN datasets.
2. Small-sample benchmark: PCN vs EInet.
3. Worst-case synthetic grids: PCN vs learned BN.
4. Random IPF scaling benchmark.


## 1) Run All 4 Experiments

From the repository root, run:

```bash
bash run_bn.sh
bash run_bn_small_samples.sh
bash run_worst_case_grid.sh
bash run_ipf.sh
```

Notes:

- These scripts already contain the paper settings (datasets, seeds, sample sizes, grid params, etc.).
- The scripts set `CUDA_VISIBLE_DEVICES` internally. Adjust GPU IDs at the top of each script if needed.
- If CUDA is unavailable, run the underlying Python files directly with CPU-compatible settings.


## 2) Generate Tables for the Paper

After experiments finish, generate LaTeX tables with:

### Experiment 1: BN benchmark (PCN vs learned BN)

```bash
python print_evaluation_bn.py
```

### Experiment 2: Small-sample benchmark (PCN vs EInet)

```bash
python print_evaluation_bn_pcn_einet.py
```

### Experiment 3: Worst-case grids

```bash
python print_evaluation_worst_case.py
```

### Experiment 4: Random IPF scaling

```bash
python print_evaluation_ipf_random.py
```


## 3) Generate the Small-Sample Plot

To generate the 1x4 figure used for the small-sample experiment:

```bash
python plot_evaluation_bn_small_samples.py
```


## 4) Output Locations

By default, outputs are stored in:

- `experiments/bn_for_workshop/`
- `experiments/bn_for_workshop_samples/`
- `experiments/bn_for_workshop_grids/`
- `experiments/ipf_random/`

Generated figures are stored in `plots/`.


## 5) Reproducibility Details

- Seed loops are already included in the run scripts.
- Results are aggregated across seeds by the `print_evaluation_*.py` scripts.