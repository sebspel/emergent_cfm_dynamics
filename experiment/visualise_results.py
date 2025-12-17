"""Generate and visualise the results of the mini-experiment"""

import math
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm.auto import tqdm

from experiment.cfm_toy_model import CFMToyModel
from experiment.utils import get_device, set_seed, load_checkpoint
from experiment.cfm_train import BASE_CHECKPOINT_PATH, sample_evenly_spaced_circle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = get_device()


@dataclass
class VisualisationConfig:
    n_particles: int = 64
    seed: int = 42
    n_steps: int = 100
    device: str = DEVICE
    model_checkpoint_path: str | Path = BASE_CHECKPOINT_PATH / "cfm_toy_model_10000.pt"


def generate_trajectories(
    model,
    device: str,
    n_particles: int = 20,
    n_steps: int = 100,
) -> np.ndarray:
    """Generate probability density paths (particles) and visualise them"""
    model_was_training = model.training
    model.eval()
    # Generate noised samples/particles
    # particles = torch.randn(
    #     (n_particles, 2),
    #     device=device,
    # )
    particles = torch.normal(
        0.0,
        0.3,
        size=(n_particles, 2),
        device=device,
    )
    time_interval = 1.0 / n_steps
    trajectories = [particles.cpu().numpy().copy()]
    with torch.no_grad():
        for step in tqdm(range(n_steps)):
            time = step * time_interval
            timesteps = torch.full(
                (n_particles, 1),
                time,
                device=device,
            )
            input_tensor = torch.cat(
                (particles, timesteps),
                dim=-1,
            )
            # Predict the velocity at the given timestep
            velocity_prediction = model(input_tensor)
            # Evolve the flow foward in time with Euler step
            particles.add_(velocity_prediction, alpha=time_interval)
            trajectories.append(particles.cpu().numpy().copy())
    model.train(model_was_training)
    return np.asarray(trajectories)


def visualise_results(trajectories: np.ndarray, n_steps: int):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    angles = np.linspace(0, 2 * math.pi, 100)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    def update(frame):
        x_coordinates = trajectories[frame, :, 0]
        y_coordinates = trajectories[frame, :, 1]
        n_particles = trajectories.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_particles))

        ax.clear()
        for i in range(n_particles):
            ax.plot(
                trajectories[: frame + 1, i, 0],
                trajectories[: frame + 1, i, 1],
                color=colors[i],
                alpha=0.4,
                linewidth=1,
            )
        ax.scatter(x_coordinates, y_coordinates, s=100, alpha=0.7)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(f"Collective Flow (t = {frame / n_steps:.2f})")

        # Target circle
        ax.plot(cos_angles, sin_angles, "r--", alpha=0.3)

    # Construct animation
    animation = FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=50,
        repeat=True,
    )
    plt.tight_layout()
    plt.show()

    # Final distribution plot
    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=150,
    )
    final_x_coordinates = trajectories[-1, :, 0]
    final_y_coordinates = trajectories[-1, :, 1]
    targets = (
        sample_evenly_spaced_circle(trajectories.shape[0], randomise=True).cpu().numpy()
    )
    ax.scatter(
        final_x_coordinates,
        final_y_coordinates,
        s=150,
        alpha=0.5,
        color="blue",
    )
    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        s=150,
        alpha=0.5,
        color="red",
        marker="x",
        label="Target Circle",
    )
    ax.plot(cos_angles, sin_angles, "k--", alpha=0.3, label="Target Spacing")
    ax.set_aspect("equal")
    ax.set_title("Final Particle Distribution (blue) vs Targets (red)")
    ax.legend()
    plt.savefig(
        "final_distribution_with_matched_samples.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def main():
    visualisation_config = VisualisationConfig()
    set_seed(visualisation_config.seed)
    model = CFMToyModel().to(device=visualisation_config.device)
    model, _ = load_checkpoint(
        model,
        visualisation_config.model_checkpoint_path,
        visualisation_config.device,
    )
    particle_trajectories = generate_trajectories(
        model,
        visualisation_config.device,
        n_particles=visualisation_config.n_particles,
        n_steps=visualisation_config.n_steps,
    )
    visualise_results(particle_trajectories, visualisation_config.n_steps)


if __name__ == "__main__":
    main()
