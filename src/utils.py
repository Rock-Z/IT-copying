import gin
import torch
import numpy as np

def load_gin_configs(gin_files: list, gin_bindings : list = []):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False
  )

@gin.configurable
def set_seed(seed : int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

