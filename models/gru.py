"""Baseline A: GRU multimodal baseline inspired by conventional pedestrian-intention models (not a PCPA reproduction).

Each input stream: Linear(F -> d) + ReLU -> GRU(hidden) -> last hidden state.
Streams are concatenated and classified by an MLP (hidden*M -> 64 -> 1). Output: logits [B].
"""
import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    def __init__(self, input_dims, d=128, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.names = list(input_dims)
        self.proj = nn.ModuleDict({k.replace(':', '_'): nn.Linear(f, d) for k, f in input_dims.items()})
        self.gru = nn.ModuleDict({k.replace(':', '_'): nn.GRU(d, hidden, num_layers, batch_first=True,
                                                             dropout=dropout if num_layers > 1 else 0.0)
                                  for k in input_dims})
        self.head = nn.Sequential(nn.Linear(hidden * len(input_dims), 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, 1))

    def forward(self, batch):
        hs = []
        for name in self.names:
            k = name.replace(':', '_')
            _, h = self.gru[k](torch.relu(self.proj[k](batch[name].float())))
            hs.append(h[-1])
        return self.head(torch.cat(hs, -1)).squeeze(-1)
