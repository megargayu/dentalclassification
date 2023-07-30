from ptflops import get_model_complexity_info

def get_macs(model):
  macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
  return macs
