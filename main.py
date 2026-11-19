from config.config import load_config
from dataset.trajectory_dataset import TrajectoryDataset
from dataset.neural_ode_dataset import NeuralODEDataset
from dataset.torch_dataset import TorchDataset
from simulation.mujoco_simulator import MujocoSimulator
from visualization.visualizer import Visualizer
from models.neural_ode import NeuralODE
from training.trainer import Trainer
from evaluation.inference import load_model
from evaluation.evaluator import evaluate
from utils.dataset_split import split_trajectory_ids
import torch
from torch.utils.data import DataLoader
import argparse
from utils.normalization import calculate_normalization_stats


def run_training(config, simulator, device):
    raw_dataset = TrajectoryDataset(config["dataset"])
    processed_dataset = NeuralODEDataset(config["dataset"])

    #Generate the datasets if they dont exist already.
    if not raw_dataset.exists():
        raw_dataset.generate(simulator=simulator)

    if not processed_dataset.exists():
        processed_dataset.generate()

    dataset_config = config["dataset"]
    trajectory_ids = list(range(dataset_config["num_trajectories"]))

    train_ids, val_ids, test_ids = split_trajectory_ids(
        trajectory_ids=trajectory_ids,
        train_ratio=dataset_config["train_ratio"],
        val_ratio=dataset_config["val_ratio"],
        seed=dataset_config["seed"]
    )

    print(f"Training trajectories:   {len(train_ids)}")
    print(f"Validation trajectories: {len(val_ids)}")
    print(f"Test trajectories:       {len(test_ids)}")

    state_std, deriv_std = calculate_normalization_stats(raw_dataset,train_ids)
    #Get standard dev. for the entire training dataset.
    print("State standard deviations:", state_std)
    print("Derivative standard deviations:", deriv_std)

    #Instantiate datasets and create dataloaders
    train_dataset = TorchDataset(dataset_config, train_ids)
    val_dataset = TorchDataset(dataset_config, val_ids)
    batch_size = config["trainer"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    model = NeuralODE(config["neural_ode"]).to(device)
    model.set_normalization_stats(state_std, deriv_std)
    model = model.to(device)
    
    trainer = Trainer(
        model=model,
        config=config["trainer"],
        device=device
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        train_time=train_dataset.t,
        val_time=val_dataset.t
    )


def run_evaluation(config, simulator, device):
    model = load_model(
        config=config,
        checkpoint_path=config["trainer"]["checkpoint_path"],
        device=device
    )
    results = evaluate(
        model=model,
        simulator=simulator,
        config=config,
        device=device,
        seed=42
    )

    #Extract simulated and predicted trajectories for comparison.
    true_trajectory = results["true_trajectory"]
    predicted_trajectory = results["predicted_trajectory"]
    metrics = results["metrics"]

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    #Plot stuff.    
    visualizer = Visualizer()
    visualizer.plot_trajectories(
        true_trajectory,
        predicted_trajectory
    )
    visualizer.plot_phase_portraits(true_trajectory, predicted_trajectory)


def main():
    parser = argparse.ArgumentParser(description="Neural ODE Project. Run with --train to train the network, with --eval to evaluate the results.")

    mode_group = parser.add_mutually_exclusive_group(required=True) #Either train or eval.

    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a new Neural ODE model"
    )
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a saved Neural ODE model"
    )

    args = parser.parse_args()

    config = load_config()

    if not config:
        print("Failed to load configuration.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    simulator = MujocoSimulator(config["simulation"])

    if args.train:
        run_training(config, simulator, device)

    elif args.eval:
        run_evaluation(config, simulator, device)


if __name__ == "__main__":
    main()