Create a CBRA sensitivity-analysis engine.

Inputs:

R12
R13
R14
R15
R16
R17
R18
R19

FedAvg baseline

FedProx baseline

Tasks:

1. Parameter sensitivity analysis

   alpha_n
   beta_q
   gamma_d

2. Contribution-weight analysis

3. Client-weight distribution analysis

4. CBRA vs FedAvg comparison

5. CBRA vs FedProx comparison

6. Statistical comparison

7. Confidence intervals

Generate:

cbra_parameter_sensitivity.csv

cbra_vs_fedavg.csv

cbra_vs_fedprox.csv

cbra_contribution_summary.csv

paper_ready_tables.csv

Figures:

alpha_n sensitivity
beta_q sensitivity
gamma_d sensitivity
weight distributions
CBRA vs baseline comparison

Rule:

Do not claim CBRA superiority unless statistical evidence supports it.

If CBRA does not outperform matched baselines,
describe CBRA as a tested heuristic.