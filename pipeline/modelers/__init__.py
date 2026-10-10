"""
The pure core of the model stage: a Bayesian ordinal Bradley-Terry model of
commander skill, fitted on multiply-imputed force data, and the WAR arithmetic
on its posterior.

Modules: ``records`` (shapes), ``design`` (raw side rows to model arrays),
``components`` (comparison graph), ``ordinal`` (numpy ordinal logit),
``priors``, ``model`` (PyMC), ``war``, ``pool`` (mixture and Rubin pooling,
convergence roll-up) and ``checks`` (dataset-level gates).

Nothing here touches the database, and nothing imports pymc or arviz at
module scope.
"""
