from dataclasses import dataclass

import torch


@dataclass
class Params:
    hidden_layer_size: int = 128
    lr: float = 0.001
    replay_buffer_size: int = 20000
    target_update_freq: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
