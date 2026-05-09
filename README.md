# yf17d-cd8-abm
Agent-based model of post-vaccination (YF-17D) CD8+ T cell dynamics
## Model structures
Three different agent-based models, using different contraction mechanisms, were created and their consistency with the experimental result was examined.
- baseline model: Only background death rates are present. No active contraction mechanism.
- density-dependent feedback (FDD): Extra cell death is induced when effector cell counts get high.
- antigen-decline-driven-contraction (ADDC): Extra cell death is induced when antigen-levels are dropping

### model calibration
The models were calibrated using experimental data generated and shared by ACTIV (VAXINFECTIO, University of Antwerp)[^refACTIV]. This data is not yet publicly available; comparison using this code cannot be done yet.


[^refACTIV]: Activ-Lab, VAXINFECTIO. Yellow fever (YF-17D) vaccination cohort dataset. Principal Investigator: B. Ogunjimi. Unpublished data, used with permission, 2025-2026.

## code files
### analyse
In the analyse.py file, the model results are compared to the experimental data. Since the latter is not yet pubicly available, this file cannot be used yet as intended.
### models_ABM
The models_ABM.py file contains the models. The model versions correspond to following models:
- v1 = baseline model
- v5 = density-dependent feedback (FDD)
- v5_2 = FDD + Tem-renewal
- v6 = antigen-decline-driven-contraction (ADDC)
### optimisation
The optimisation.py file contains the code necessary to calibrate the model parameters. This file requires the experimental data too (not yet available).
### sensitivity_analysis
Sensitivity analysis of the optimal parameters. File cannot be used yet, since the experimental data (which determines the RMSE) is not available.
## exploratory files
### model_compartments
This file contains an ODE model of post-vaccination CD8 T cell dynamics. The model was made for exploration, not used for actual analysis. The file is kept for reference and to show the initial ideas for modelling the T cell response.