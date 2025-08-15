# Universal script to launch experiments:
# - loads config
# - instantiates dataset via dataset_factory
# - instantiates model via model_factory
# - trains using LightningModule
# - saves checkpoints and metrics