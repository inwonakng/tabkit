import os
import torch
import random
import numpy as np


# borrowed from https://github.com/jyansir/tp-berta/tree/main/scripts/finetune/default/run_default_config_tpberta.py
def set_random_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
