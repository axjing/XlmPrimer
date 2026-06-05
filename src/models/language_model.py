import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configuration_model import GPTConfig
from models.layers import RMSNorm

class LanguageModelMLP(nn.Module):
    def __init__(self, cfg:GPTConfig) -> None:
        super().__init__()
        
        
class LanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()