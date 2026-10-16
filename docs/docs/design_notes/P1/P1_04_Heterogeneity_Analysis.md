Build a federated heterogeneity analysis module.

Input:
   patient partition assignment

For every client compute:

1. image count
2. patient count
3. label prevalence

Compare each client against global distribution.

Metrics:

1. prevalence shift
2. Jensen-Shannon divergence
3. mean prevalence difference

Produce:

heterogeneity_summary.csv
heterogeneity_visualizations.png

Provide publication-ready tables.
`