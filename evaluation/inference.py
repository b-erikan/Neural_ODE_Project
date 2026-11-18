import torch
from pathlib import Path
from models.neural_ode import NeuralODE


def load_model(checkpoint_path, config, device):

    model = NeuralODE(config["neural_ode"])

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found.")

    # Load weights as state dictionary. Then recreate the same model from that.
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict_trajectory(model, x0, t, device):

    x0 = torch.as_tensor(
        x0,
        dtype=torch.float32,
        device=device
    )

    t = torch.as_tensor(
        t,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        prediction = model.integrate(x0, t)

    return prediction