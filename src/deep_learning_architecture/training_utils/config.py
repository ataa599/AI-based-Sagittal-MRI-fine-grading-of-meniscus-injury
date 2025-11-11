import os
import random
import numpy as np
import torch


class DenseNetSagittalConfig:
    def __init__(self):
        # 1. Basic Settings (tuned)
        self.seed = 42
        self.set_determinism(self.seed)
        # tuned hyperparams:
        self.learning_rate   = 0.00014287398259465764
        self.weight_decay    = 0.00024959381702548865
        self.label_smoothing = 0.0450847234574019
        self.use_pretrained  = False
        self.chosen_variant  = '121'
        self.optimizer_name  = 'AdamW'
        self.scheduler_name  = 'multistep'
        self.ms_gamma        = 0.09363595088672585
        self.ms_m1_fraction  = 0.5680749027320107
        self.ms_m2_fraction  = 0.8856154283737834

    # =======================
    # Determinism helper
    # =======================
    def set_determinism(self, seed: int = 42):
        # IMPORTANT: set before any CUDA work; safe to call in main() before model/tensors
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS paths

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # No TF32 (keep strict FP32 to avoid minor numeric drift)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Turn off autotune & force deterministic kernels
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
