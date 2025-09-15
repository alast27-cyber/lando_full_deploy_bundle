import torch
from app.model import LandoNet

def test_forward_shape():
    model = LandoNet(16)
    x = torch.randn(2,16)
    recon, cscore = model(x)
    assert recon.shape == (2,16)
    assert cscore.shape[0] == 2
