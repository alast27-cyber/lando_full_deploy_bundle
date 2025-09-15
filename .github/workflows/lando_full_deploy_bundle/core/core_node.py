import math
import torch
import random
from typing import Dict, List

class CoreNode:
    def __init__(self, model_class, input_dim, base_width=64, grow_factor=2,
                 min_width=16, max_width=512, patience=5, grow_threshold=0.98, prune_threshold=1.02):
        self.model_class = model_class
        self.input_dim = input_dim
        self.base_width = base_width
        self.grow_factor = grow_factor
        self.min_width = min_width
        self.max_width = max_width
        self.patience = patience
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.history = []
        self.modulators = None

    def observe(self, val_loss: float):
        self.history.append(val_loss)
        if len(self.history) > 2*self.patience:
            self.history = self.history[-2*self.patience:]
        action = "none"
        if len(self.history) >= self.patience:
            recent = sum(self.history[-self.patience:]) / self.patience
            prev = sum(self.history[-2*self.patience:-self.patience]) / self.patience if len(self.history) >= 2*self.patience else None
            if prev is not None:
                ratio = recent / prev if prev > 0 else 1.0
                if ratio < self.grow_threshold and self.base_width * self.grow_factor <= self.max_width:
                    self.base_width = int(min(self.max_width, self.base_width * self.grow_factor))
                    action = "grow"
                elif ratio > self.prune_threshold and self.base_width // self.grow_factor >= self.min_width:
                    self.base_width = int(max(self.min_width, self.base_width // self.grow_factor))
                    action = "prune"
        self._update_modulators()
        return action

    def _update_modulators(self):
        n_linear = 11
        rng = random.Random(int(self.base_width))
        modulators = {}
        for i in range(n_linear):
            base = 1.0 - (self.base_width - self.min_width) / max(1.0, (self.max_width - self.min_width)) * 0.3
            m = base * (1.0 + (rng.random() - 0.5) * 0.4)
            m = max(0.1, min(2.0, m))
            modulators[i] = m
        self.modulators = modulators

    def apply_modulation_to_grads(self, model: torch.nn.Module):
        if self.modulators is None:
            return
        linear_modules = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        for idx, lin in enumerate(linear_modules):
            mod = self.modulators.get(idx, 1.0)
            for p in lin.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(mod)

    def rebuild_and_transfer(self, old_state: Dict[str, torch.Tensor], new_base_width: int, model_builder):
        new_model = model_builder(self.input_dim, base_width=new_base_width, grow_factor=self.grow_factor)
        new_state = new_model.state_dict()
        for name, tensor in old_state.items():
            if name not in new_state:
                continue
            old_t = tensor
            new_t = new_state[name]
            if old_t.shape == new_t.shape:
                new_state[name] = old_t.clone()
                continue
            if old_t.ndim == 2 and new_t.ndim == 2:
                min0 = min(old_t.shape[0], new_t.shape[0])
                min1 = min(old_t.shape[1], new_t.shape[1])
                new_slice = new_t.clone()
                new_slice[:min0, :min1] = old_t[:min0, :min1]
                new_state[name] = new_slice
            elif old_t.ndim == 1 and new_t.ndim == 1:
                m = min(old_t.shape[0], new_t.shape[0])
                new_slice = new_t.clone()
                new_slice[:m] = old_t[:m]
                new_state[name] = new_slice
        new_model.load_state_dict(new_state)
        return new_model
