# Universal factory: takes architecture config and returns model instance
# Supports:
# - single networks (SIREN, FourierNet)
# - multi-network systems (Neural DMD)
# - hypernetworks
# Handles nested configs and ensures modularity