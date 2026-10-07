"""
Multiple imputation of missing log troop totals for the impute stage.

One missing variable (log troops) and everything else a covariate, so chained
equations reduce to a single conditional model; it is fitted jointly instead,
marginalising each battle's shared effect analytically so two-sided missingness
needs no special case. See :mod:`pipeline.imputers.model`.

Modules: ``records`` (shapes), ``priors`` (config and era bins), ``load``
(database reads), ``likelihood`` (reconcile output to likelihood, with the
prior divided out), ``design`` (aligned arrays), ``model`` (PyMC),
``conditional`` (draws per round), ``summarise``, ``gates``, ``outputs``
(files) and ``store`` (``missing_data_log`` writes).

Nothing here imports pymc at module scope.
"""
