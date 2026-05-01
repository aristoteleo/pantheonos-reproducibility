# PantheonOS Reproducibility

This repository contains reproduction materials (scripts, intermediate results, reports) for the figures in the **PantheonOS** paper.

The actual reasoning trajectories the agent took to produce each figure are recorded as **replayable bundles** hosted on Hugging Face. Click any "▶ Replay" link below to step through the exact agent conversation, tool calls, and intermediate outputs in your browser via the public PantheonOS replay viewer.

Heavy data files (`.h5ad`, etc.) are hosted on **Zenodo** (link to be added once uploaded).

---

## Figures → Trajectories / Reports

### Figure 2 — Immune Oncology Gene Panel Design

A 1000-plex human immune-oncology gene panel designed by multi-agent collaboration. The team downloads CELLxGENE Census data, runs HVG/DE/RF/scGeneFit/SpaPROS gene selection algorithms, benchmarks panel quality, and produces a final annotated panel with LaTeX report.

- ▶ [Replay trajectory](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fimmune_oncology_gene_panel_design.zip)

### Figure 3 — Evolution Reports (algorithm self-improvement)

Pantheon's evolution system iteratively improves batch-correction algorithms. Static HTML reports of the evolution runs:

- 📄 [BBKNN evolution](https://pantheonos.stanford.edu/evolve/bbknn.html)
- 📄 [Harmonypy evolution](https://pantheonos.stanford.edu/evolve/harmonypy.html)
- 📄 [Scanorama evolution](https://pantheonos.stanford.edu/evolve/scanorama.html)

### Figure 4 — *(coming soon)*

### Figure 5 — 3D Human Fetal Heart Multi-Omic Analysis

A four-step analysis workflow on the PCW12 human fetal heart MERFISH + scRNA + snATAC dataset.

| Step | Trajectory |
| --- | --- |
| 5.1 Heart MERFISH data overview | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fheart_merfish_data_overview.zip) |
| 5.2 3D human fetal heart disease gene pattern | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fhuman_heart_3d_disease_gene_pattern.zip) |
| 5.3 Spatial ligand-receptor disease analysis | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fspatial_ligand_receptor_disease.zip) |
| 5.4 ATAC spatial mapping via MOSCOT | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fatac_spatial_mapping_moscot.zip) |

Local working files for Figure 5 live under [`human-heart/`](human-heart/) — `PCW7_analysis/` and `PCW12_analysis/` contain scripts, figures, and reports. Raw `.h5ad` inputs are on Zenodo.

### Figure 6 — *(coming soon)*

---

## Repository layout

```
human-heart/                # Figure 5 working directory
├── PCW7_analysis/          # Earlier developmental stage (scripts, results)
├── PCW12_analysis/         # Main figure-5 analyses
│   ├── scripts/            # Reproducible Python scripts
│   ├── figures/            # Final figure PNGs/PDFs
│   ├── REPORT.md           # Auto-generated analysis report
│   └── REPORT_ATAC.md      # ATAC sub-analysis report
├── data/                   # Raw .h5ad inputs (Zenodo, gitignored)
└── references/

mouse-embryo-e6/            # In progress
```

The full Pantheon project state (chat memory, agent runtime, virtual envs) is gitignored — see `.gitignore`. To re-run any analysis end-to-end, replay the corresponding trajectory in PantheonOS Desktop or via the web replayer.

---

## Data hosting

| Asset | Where |
| --- | --- |
| Replayable agent trajectories | [`NaNg/Pantheon-Agent-Trajectory`](https://huggingface.co/datasets/NaNg/Pantheon-Agent-Trajectory) on Hugging Face |
| Raw `.h5ad` data (heart, mouse embryo) | Zenodo (DOI to be added) |
| Evolution HTML reports | https://pantheonos.stanford.edu/evolve/ |

---

## License

See [LICENSE](LICENSE).
