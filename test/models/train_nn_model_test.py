import subprocess
from pathlib import Path

model_nn = Path("models/model_nn.keras")
model_nn.resolve()


def test_train_nn_model_save_a_file_with_model():
    subprocess.run(args=["make", "train_nn"])
    assert model_nn.is_file()
