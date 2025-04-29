# IntelliLight 4-Phase Extension

This codebase is adapted from [Wei et al. (IntelliLight)](https://dl.acm.org/doi/10.1145/3219819.3220096), originally designed for 2-phase reinforcement learning traffic signal control. The original code was developed in Python 3.6 with TensorFlow 1.9 and SUMO 0.32; we ported it to Python 3.8, TensorFlow 2.12 and SUMO 1.22, with code adjustments to ensure compatibility and stable runtime.

## Key Modifications

- **4-Phase Extension**:  
  We extend the original 2-phase logic (NS and EW straight-only phases) to a 4-phase setup:  
  1. NS straight + left  
  2. SN straight + left  
  3. WE straight + left  
  4. EW straight + left
 
  Only one directional movement is allowed per phase for safety and clarity in evaluation.

- **Reward Function**:  
  We modified and tuned the reward function to better capture waiting time, stops, and system congestion under this expanded control scheme.

- **Configuration**:  
  We use our own config files based on `config2-4` in the original paper, modifying traffic flow to include left turns and limiting simulation time to 7200s for manageable run time.

## Code Overview

- `agent.py`:  
  Base class for RL agents, defines common structure for handling state, memory, and interaction.

- `deeplight_agent.py`:  
  Main implementation of the IntelliLight Deep Q-Learning agent, adapted for 4-phase logic.

- `network_agent.py`:  
  Defines shared and separate neural network branches, including convolutional layers and Q-value outputs.

- `sumo_agent.py`:  
  Manages the interface between SUMO and the RL agent. Handles simulation stepping, vehicle state updates, and reward calculations.

- `map_computor.py`:  
  Parses the SUMO network, maps phases to affected lanes, builds spatial traffic features, and computes lane-based metrics.

- `traffic_light_dqn.py`:  
  Core training logic: initializes agents, manages pretraining, action selection, Q-network updates, and memory logging.

- `runexp.py`:  
  Entry point for launching experiments. Loads configurations, prepares training runs, and executes the full pipeline.
---

## How to Run

To run the full training and evaluation pipeline:

```bash
python runexp.py


