"""Single-model UQ methods that give the MLP backbone a predictive distribution.

One module per method, holding the :class:`~admet_uq.heads.mlp.Arm` that reads
each epoch's weights and the head the registry reaches by name. Each records a
location and a spread, which is what separates these from the bare point
predictor and from the calibrators, which reshape stored predictions instead.
"""
