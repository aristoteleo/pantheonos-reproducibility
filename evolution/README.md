# Pantheon Evolution Experiments

Original run scripts and recorded outputs for the **algorithmic self-improvement** experiments in the PantheonOS paper (Figures 2 and 3).

Each experiment uses `pantheon.evolution` (LLM-guided code mutations + MAP-Elites quality-diversity search) to evolve a target algorithm against a fitness evaluator.

The full HTML evolution reports (40+ MB each) are not committed here; they are hosted at:
- https://pantheonos.stanford.edu/evolve/bbknn.html
- https://pantheonos.stanford.edu/evolve/harmonypy.html
- https://pantheonos.stanford.edu/evolve/scanorama.html

What lives in this tree are the inputs (initial code, evaluator, run script) and the trajectory artifacts in JSON form (`evolution_report.json`, `evolution_state.json`, `island_coordinates.json`, `metadata.json`, `score_history_fixed.json`), the evolved code variants, and the figures used in the paper.

## Layout

```
evolution/
├── batch_correction/                    # Figure 3
│   ├── metrics.py                       # shared SI / mixing / convergence metrics
│   ├── plot_paper_figures.py            # cross-method paper figure generation
│   ├── bbknn/
│   │   ├── bbknn/                       # initial implementation (evolution target)
│   │   ├── evaluator.py
│   │   ├── run_evolution.py
│   │   └── runs/                        # evolution outputs (renamed from results/)
│   │       ├── config.yaml, metadata.json, evolution_report.json, ...
│   │       ├── bbknn_optimized/         # final selected variant
│   │       ├── bbknn_balanced/, bbknn_fast/, bbknn_similar_speed/  # other Pareto picks
│   │       └── paper_figures*/          # PDF/PNG used in figure 3
│   ├── harmonypy/
│   │   ├── harmony.py                   # initial implementation
│   │   ├── evaluator.py, evaluator_pbmc.py
│   │   ├── benchmark_{final,pancreas}.py
│   │   ├── run_evolution.py
│   │   ├── run_evolution_claude_model.py    # secondary run with Claude as the mutator
│   │   ├── runs/                        # primary evolution run
│   │   │   └── harmony_{272,best_mixing,claude,optimized}.py  # selected variants
│   │   └── runs_claude_model/           # secondary run
│   └── scanorama/
│       ├── scanorama/                   # initial implementation
│       ├── annoy.py
│       ├── evaluator.py
│       ├── run_evolution.py
│       └── runs/
│           ├── scanorama_optimized/
│           └── paper_figures{_pancreas,_pbmc,_tma,_ffdaa1f0}/
└── gene_panel/                          # Figure 2 — RL gene-panel selection
    ├── README.md
    ├── rl_gene_panel.py                 # initial RL implementation
    ├── evaluator.py
    ├── run_evolution.py
    ├── evaluate_gene_panel.py
    ├── scripts/run_preprocessing.py
    ├── runs_explore_primary/            # exploration-mode evolution
    └── runs_optimize_primary/           # optimization-mode evolution
```

## Reproducing a run

The evolution scripts depend on `pantheon.evolution`, the runtime in the main
[PantheonOS repository](https://github.com/aristoteleo/PantheonOS). With that
installed, each method is self-contained:

```bash
cd evolution/batch_correction/harmonypy
python run_evolution.py --iterations 100 --output runs_new/
```

Input data CSVs (PBMC, pancreas, TMA, ffdaa1f0) are not committed — they live
under `evolution/batch_correction/data/` in the source repo and are linked from
the corresponding Zenodo deposition for the paper.
