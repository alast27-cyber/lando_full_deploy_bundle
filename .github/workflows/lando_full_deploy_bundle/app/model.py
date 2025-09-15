import torch
import torch.nn as nn

class LandoNet(nn.Module):
    def __init__(self, input_dim, base_width=64, grow_factor=2, freeze_layers=None):
        super().__init__()
        widths = []
        w = base_width
        for i in range(5):
            widths.append(w)
            w = int(w * grow_factor)
        for i in range(5):
            widths.append(widths[4 - i])
        self.widths = widths
        layers = []
        in_dim = input_dim
        for idx, out_dim in enumerate(widths):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.freeze_layers = set(freeze_layers or [])
        if self.freeze_layers:
            self._apply_freeze()
        self.contradiction_layer = nn.Sequential(
            nn.Linear(input_dim, max(16, input_dim//4)),
            nn.ReLU(),
            nn.Linear(max(16, input_dim//4), 1),
            nn.Sigmoid()
        )

    def _apply_freeze(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for idx in self.freeze_layers:
            if 0 <= idx < len(linear_layers):
                for p in linear_layers[idx].parameters():
                    p.requires_grad = False

    def forward(self, x):
        recon = self.net(x)
        cscore = self.contradiction_layer(x)
        return recon, cscore.squeeze(-1)
