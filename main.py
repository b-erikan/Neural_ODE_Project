from config.config import load_config

from dataset.trajectory_dataset import TrajectoryDataset
from dataset.neural_ode_dataset import NeuralODEDataset
from simulation.mujoco_simulator import MujocoSimulator
from visualization.visualizer import Visualizer


def main():

    config = load_config()

    if not config:
        print("Failed to load configuration.")
        return

    simulator = MujocoSimulator(config["simulation"])
    raw_dataset = TrajectoryDataset(config["dataset"])
    if not raw_dataset.exists():
        raw_dataset.generate(simulator=simulator)
    
    #processed_dataset = NeuralODEDataset(config["dataset"])
    visualizer = Visualizer()
    visualizer.visualize(raw_dataset)
    print("Dataset generation completed.")



if __name__ == "__main__":
    main()