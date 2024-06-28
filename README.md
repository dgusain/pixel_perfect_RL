# Pixel Perfect: Accelerating AI Mastery in Atari with Advanced Convolutional Reinforcement Learning

## Project Overview

This project explores and enhances the efficacy of Convolutional Neural Networks (CNNs) in conjunction with Reinforcement Learning (RL) techniques, such as Deep Q-Networks (DQN), Double DQNs, and Proximal Policy Optimization (PPO), to train autonomous agents to play various Atari games. Starting with the relatively simple environment of Pong and progressing to more complex challenges such as Breakout and Montezuma's Revenge, the project aims to fine-tune hyperparameters that accelerate the learning process. The ultimate goal is to develop faster, more efficient algorithms capable of quickly training RL agents without resorting to binary frame conversion, thereby pushing the boundaries of how raw pixel data can be leveraged to teach machines to play—and excel at—video games.

## Abstract

As artificial intelligence progresses, teaching machines to understand and interact with complex environments has profound implications, not only for gaming but for broader applications in robotics, autonomous navigation, and decision-making systems. This project focuses on applying advanced CNNs and RL techniques to the task of training agents on Atari games, using the games as benchmarks for developing faster and more efficient learning algorithms.

## Objectives

1. **Integrate and Optimize CNNs with RL Techniques**: Evaluate and refine the integration of CNNs with RL algorithms such as DQN, Double DQN, and PPO to train agents on Atari games, focusing on hyperparameter optimization for efficient learning.
2. **Scale Learning from Simple to Complex Tasks**: Progress agent training from the basic game of Pong to the intricate environments of Breakout and Montezuma's Revenge, aiming to enhance the agents' ability to learn and strategize from complex visual inputs.
3. **Investigate the Transferability of Game-Learned Skills**: Explore the potential for strategies and recognition patterns learned in game environments to be transferred to real-world applications, such as autonomous navigation and computer vision tasks.

## Algorithms Applied

### Deep Q-Network (DQN)

An approach in reinforcement learning that combines Q-learning with deep neural networks to handle high-dimensional state and action spaces.

**Merits:**
- Uses experience replay for sample efficiency.
- Freezes target Q-network to stabilize learning.
- Can handle large and complex state spaces.

**Demerits:**
- Non-linear function approximators can lead to instability.
- Requires significant computational resources.
- Sensitive to hyperparameters.

### Double Deep Q-Network (Double DQN)

An extension of DQN that reduces the overestimation bias of action values by using two networks: policy and target networks.

**Merits:**
- More accurate value estimates by reducing overestimation bias.
- Better policy performance and more stable learning.

**Demerits:**
- Increased computational overhead and memory usage.
- Complexity in updating and managing two networks.

### Proximal Policy Optimization (PPO)

A policy gradient method that optimizes a surrogate objective function via stochastic gradient descent for stable and efficient training.

**Merits:**
- More stable and robust to hyperparameter settings.
- Easier to implement and tune compared to other policy gradient methods.
- More sample efficient.

**Demerits:**
- Computationally expensive due to multiple epochs of minibatch updates.
- May be inefficient in terms of sample reuse as an on-policy algorithm.

## Environments

- **Pong-v4**: Classic Atari 2600 game implemented in OpenAI Gym.
- **PongDeterministic-v4**: Deterministic setting of Pong with reduced randomness.
- **BreakoutNoFrameskip-v4**: Atari game where players control a paddle to break bricks. No frame skipping provides granular control.
- **GravitarDeterministic-v4**: Navigate a spaceship through challenging planetary systems in a deterministic setting.
- **LunarLander-v4**: Control a lunar lander to land safely on the moon, simulating physics like gravity and thrust.

## Results

### Pong-v4 DQN

- **Inference**: Gradual improvement over 9000 episodes with significant variability, indicating some instability.

### Pong-v4 Double DQN

- **Inference**: Improved and more stable performance over 5000 episodes compared to standard DQN.

### PongDeterministic-v4 PPO

- **Inference**: Near-optimal performance with efficient exploitation of learning signals and high evaluation scores.

### GravitarDeterministic-v4 PPO

- **Inference**: Steady increase in rewards indicating effective learning, with performance stabilizing after initial learning phase.

### BreakoutNoFrameskip-v4 PPO

- **Inference**: Significant learning observed with an upward trend in rewards, suggesting effective strategy learning.

### Lunar-Lander-v2 PPO (Playground)

- **Inference**: Steady improvement and possible convergence to a stable policy within 2000 episodes.

## References

1. [GitHub - OpenAIPong-DQN](https://github.com/bhctsntrk/OpenAIPong-DQN)
2. [Deep Q-Network (DQN)-I: OpenAI Gym Pong and Wrappers](https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af)
3. [OpenAI Baselines: DQN](https://openai.com/research/openai-baselines-dqn)
4. [GitHub - DQN-Atari-Breakout](https://github.com/GiannisMitr/DQN-Atari-Breakout)
5. [Learning Montezuma’s Revenge from a single demonstration](https://openai.com/research/learning-montezumas-revenge-from-a-single-demonstration)
6. [CleanRL PPO Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

## How to Run

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dgusain/pixel_perfect_RL.git
   cd pixel_perfect_RL
