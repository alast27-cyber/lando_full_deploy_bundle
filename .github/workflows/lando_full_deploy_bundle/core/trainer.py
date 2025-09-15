import os, json, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from app.model import LandoNet
from core.core_node import CoreNode

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "example_pairs.json")

def build_dataset(pairs):
    texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    vect = CountVectorizer(max_features=1024, binary=True)
    vect.fit(texts)
    X_in = vect.transform([p[0] for p in pairs]).toarray().astype(float)
    Y_out = vect.transform([p[1] for p in pairs]).toarray().astype(float)
    return X_in, Y_out, vect

def evaluate(model, X_val, Y_val, device='cpu'):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        xb = torch.tensor(X_val, dtype=torch.float32, device=device)
        yb = torch.tensor(Y_val, dtype=torch.float32, device=device)
        recon, cscore = model(xb)
        loss = criterion(recon, yb).item()
    return loss

def train_with_core(pairs=None, epochs=40, device='cpu'):
    if not pairs:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r') as f:
                pairs = json.load(f)
        else:
            pairs = [['hello','hi there']]
    X, Y, vect = build_dataset(pairs)
    input_dim = X.shape[1]
    core = CoreNode(LandoNet, input_dim, base_width=64, grow_factor=2)
    model = LandoNet(input_dim, base_width=core.base_width, grow_factor=core.grow_factor).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    idx = list(range(len(X)))
    random.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))
    train_idx = idx[:split]
    val_idx = idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx] if len(val_idx)>0 else (X[train_idx], Y[train_idx])
    batch_size = 4
    for epoch in range(epochs):
        model.train()
        perm = list(range(len(X_train)))
        random.shuffle(perm)
        for i in range(0, len(perm), batch_size):
            ids = perm[i:i+batch_size]
            xb = torch.tensor(X_train[ids], dtype=torch.float32, device=device)
            yb = torch.tensor(Y_train[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            recon, cscore = model(xb)
            loss_recon = criterion(recon, yb)
            loss = loss_recon + 0.1 * torch.mean(cscore)
            loss.backward()
            core.apply_modulation_to_grads(model)
            optimizer.step()
        val_loss = evaluate(model, X_val, Y_val, device=device)
        action = core.observe(val_loss)
        print(f"Epoch {epoch+1}/{epochs} val_loss={val_loss:.6f} action={action} base_width={core.base_width}")
        if action in ('grow','prune'):
            old_state = model.state_dict()
            new_model = core.rebuild_and_transfer(old_state, core.base_width, lambda in_dim, base_width, grow_factor: LandoNet(in_dim, base_width=base_width, grow_factor=grow_factor))
            model = new_model.to(device)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lando_phase2.pt')
    with open('models/vocab.json','w') as f:
        json.dump({'vocab': vect.get_feature_names_out().tolist()}, f)
    return model, vect

if __name__ == '__main__':
    train_with_core(epochs=5, device='cpu')
