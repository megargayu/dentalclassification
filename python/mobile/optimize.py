import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

def optimize_model(model):
  example = torch.rand(1, 3, 224, 224)

  model = model.cpu()
  traced_script_module = torch.jit.trace(model, example)
  traced_script_module_optimized = optimize_for_mobile(traced_script_module)
  
  return traced_script_module_optimized

def save_mobile_model(traced_script_module_optimized, path):
  traced_script_module_optimized._save_for_lite_interpreter(path)
