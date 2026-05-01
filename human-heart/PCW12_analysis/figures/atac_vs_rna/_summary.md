# ATAC vs RNA comparison — summary

- shared genes: 215
- highlight genes: 22 (ACTA2, ACTC1, ACTN2, CHD4, CHD7, DLL4, ELN, FBN1, GATA4, GATA6, KMT2D, MEIS2, MYBPC3, MYL2, MYL3, TBX1, TBX20, TNNC1, TNNI3, TNNT2, TPM1, VCL)
- median per-gene Pearson r (across 100k cells): 0.129
- mean per-gene Pearson r: 0.217
- top 5 concordant genes: [{'gene': 'RBM20', 'pearson_r': 0.8678043854385207}, {'gene': 'FHOD3', 'pearson_r': 0.8528223848535017}, {'gene': 'LDB3', 'pearson_r': 0.8147256032751136}, {'gene': 'CORIN', 'pearson_r': 0.8093289515392468}, {'gene': 'MYBPC3', 'pearson_r': 0.8063505614975306}]
- bottom 5 (most divergent) genes: [{'gene': 'MRPL44', 'pearson_r': -0.1053376388905087}, {'gene': 'CHD4', 'pearson_r': -0.11084063639076308}, {'gene': 'RPL5', 'pearson_r': -0.1550111343693279}, {'gene': 'CFC1', 'pearson_r': nan}, {'gene': 'NKX2-6', 'pearson_r': nan}]

## Per-trait

| Trait | n_genes | pearson_r | atac_mean_z | rna_mean_z |
|---|---|---|---|---|
| HypertrophicCardiomyopathy | 27 | 0.793 | 0.000 | 0.000 |
| DilatedCardiomyopathy | 40 | 0.725 | -0.000 | 0.000 |
| Familial_thoracic_aortic_aneurysm_and_aortic_dissection | 9 | 0.418 | 0.000 | 0.000 |
| Valve_defects | 74 | 0.405 | 0.000 | 0.000 |
| Malformation_of_the_outflow_tract | 57 | 0.379 | -0.000 | 0.000 |
| Atrial_septal_defect | 87 | 0.377 | 0.000 | 0.000 |
| Atrioventricular_septal_defect | 22 | 0.364 | 0.000 | 0.000 |
| Single_ventricle_disease | 17 | 0.363 | -0.000 | 0.000 |
| PCGC_DeNovoVariants | 57 | 0.356 | 0.000 | 0.000 |
| Ventricular_septal_defect | 33 | 0.336 | 0.000 | 0.000 |
