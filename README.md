# Code for "How Deep Are Deep GPs, Really?"

Code accompanying

> Mark Kozdoba and Shie Mannor. *How Deep Are Deep GPs, Really? A Sharp Threshold
> and a Non-Gaussian Limit for Compositional GPs.* NeurIPS 2026.
> arXiv: https://arxiv.org/abs/2606.08218 · OpenReview: https://openreview.net/forum?id=F6fd33khoH

Running the four notebooks below from a fresh clone produces every figure that
appears in the paper.

## Files

- **`chain.py`** — Simulation library. Provides the critical-radius helper
  `r_c_of_d`, the supercritical-rate helper `mu_of`, the Jensen-bound helper
  `jensen_L_star`, the scalar pairwise-distance chain (`step_log`,
  `simulate_L_trajectories`, `simulate_L_long`), and the position chain
  `simulate_Z_chain`. All experiments derive from these.

- **`01_trajectories.ipynb`** — Produces every trajectory- and stationary-law
  figure in the paper:
  - `figs/trajectories_stationary_d100.pdf` (Figure 2; three-pane $d = 100$
    summary of the dichotomy)
  - `figs/trajectories_above.pdf` (Figure 4; supercritical trajectories at
    $d = 1, 10, 100$)
  - `figs/trajectories_below.pdf` (Figure 5; subcritical trajectories at
    $d = 1, 10, 100$, ten $\lambda$ values)
  - `figs/stationary_multidim.pdf` (Figure 6; empirical stationary law of
    $\log v_i^2$ at $d = 1, 10, 100$)
  - `figs/trajectories_std.pdf` (Figure 7; SEM of the mean $\log v_i^2$
    estimator across replicates)

  Approximate runtime: a few minutes on a single CPU core.

- **`02_perchain_d1_d10.ipynb`** — Produces the per-chain visualisation figures
  at low and moderate ambient dimension:
  - `figs/pi_Z_hist_d1.pdf` (Figure 8; $d = 1$, per-chain histograms)
  - `figs/pi_Z_pca_d10.pdf` (Figure 9; $d = 10$, per-chain PC1--PC2)
  - `figs/pi_Z_tsne_d10.pdf` (Figure 11; $d = 10$, per-chain t-SNE)

  Approximate runtime: 5--10 minutes on a single CPU core.

- **`03_perchain_d100.ipynb`** — Produces the two main-paper $d = 100$
  visualisation figures at depth $i = 1000$:
  - `figs/pi_Z_tsne_d100.pdf` (Figure 3; main-paper t-SNE)
  - `figs/pi_Z_pca_d100.pdf` (Figure 12; PCA companion)

  Approximate runtime: 30--40 minutes on a single CPU core (heaviest of the
  four; $5$ chains $\times 3$ $\lambda$ values $\times 1000$ Cholesky
  factorisations of a $1000 \times 1000$ correlation matrix).

- **`04_pca_density.ipynb`** — Produces the empirical PC1--PC2 density
  companions at $d = 10$ and $d = 100$:
  - `figs/pca_density_d10.pdf` (Figure 10; density companion to Figure 9)
  - `figs/pca_density_d100.pdf` (Figure 13; density companion to Figure 12)

  Uses the same per-$\lambda$ seeds as `02` (for $d = 10$) and `03` (for
  $d = 100$), so the projections are taken from the same chain realisations
  as the corresponding scatter figures. Approximate runtime: 5--10 minutes.

## Running

```bash
pip install numpy scipy matplotlib scikit-learn jupyter nbformat
cd supplementary_code/
jupyter notebook 01_trajectories.ipynb         # or jupyter lab, or VSCode
```

Each notebook saves its figures to `figs/` (a subdirectory of this code
directory). The figures rendered by the notebooks are exactly the figures
appearing in the paper. Total wall-clock for all four notebooks is well
within an hour on a single CPU core.

## Reproducibility

All randomness is seeded with `numpy.random.default_rng(...)`; reruns
reproduce the same figures bit-for-bit modulo numpy / sklearn version
differences.

## Citation

```bibtex
@inproceedings{kozdoba2026howdeep,
  title     = {How Deep Are Deep {GP}s, Really? A Sharp Threshold and a Non-{G}aussian Limit for Compositional {GP}s},
  author    = {Kozdoba, Mark and Mannor, Shie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```
