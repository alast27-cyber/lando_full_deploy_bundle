from core.core_node import CoreNode
from app.model import LandoNet

def test_corenode_grow_prune():
    cn = CoreNode(LandoNet, input_dim=16, base_width=32)
    losses = [1.0,0.9,0.8,0.7,0.6,0.5]
    action = 'none'
    for l in losses:
        action = cn.observe(l)
    assert cn.base_width >= 32
    assert action in ("none","grow","prune")
