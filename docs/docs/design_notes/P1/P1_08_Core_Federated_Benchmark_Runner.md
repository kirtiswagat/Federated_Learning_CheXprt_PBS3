Create a federated experiment runner.

Support runs:

R01:
FedAvg IID 5 clients 10 rounds

R02:
FedAvg Dirichlet alpha=0.3

R03:
FedAvg Dirichlet alpha=0.1

R04:
FedProx IID
mu=0.001

R05:
FedProx Dirichlet alpha=0.1
mu=0.001

Use:

ResNet50
AdamW
batch size 16
one local epoch

Save:

history.csv
predictions.csv
checkpoints
ledger records

Evaluate after each round.