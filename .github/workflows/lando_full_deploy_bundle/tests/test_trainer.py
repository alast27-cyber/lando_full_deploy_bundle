from core.trainer import train_with_core

def test_trainer_runs(tmp_path):
    model, vect = train_with_core(epochs=1)
    assert model is not None
    assert len(vect.get_feature_names_out()) > 0
