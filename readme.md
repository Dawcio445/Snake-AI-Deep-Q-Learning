# Snake AI - Deep Reinforcement Learning

Artificial intelligence agent learning to play Snake using Deep Q-Network (DQN), PyTorch, and reinforcement learning techniques.

The project focuses on building an autonomous agent capable of learning optimal gameplay strategies through interaction with the environment and continuous improvement based on collected experience.

---

## Overview

This project implements a reinforcement learning agent that learns to play the classic Snake game without predefined movement rules.

The agent observes the current game state, selects actions using a neural network, receives rewards based on its performance, and improves its strategy through thousands of training iterations.

The goal of the project was to explore practical applications of deep learning, reinforcement learning algorithms, and GPU-accelerated model training.

---

## Features

- Deep Q-Network (DQN)
- Experience Replay
- Prioritized Experience Replay (PER)
- Dueling Deep Q-Network architecture
- Epsilon-Greedy exploration strategy
- Soft Target Network Updates
- GPU accelerated training
- Custom reinforcement learning environment
- Neural network based decision making
- Autonomous gameplay

---

## How It Works

The agent learns through interaction with the Snake environment:

1. The current game state is converted into input data.
2. The neural network predicts the expected value of available actions.
3. The agent selects an action using an exploration strategy.
4. The environment returns:
   - reward,
   - next state,
   - game status.
5. The experience is stored in replay memory.
6. The neural network is updated using sampled experiences.

Over time, the agent improves its ability to survive and achieve higher scores.

---

## Implemented Algorithms

### Deep Q-Network (DQN)

A neural network approximates the Q-function, allowing the agent to estimate the expected reward for each possible action.

### Prioritized Experience Replay (PER)

Instead of sampling experiences randomly, important transitions with higher learning value are selected more frequently.

Benefits:

- Faster learning process
- Better utilization of important experiences
- Improved training stability

### Dueling Deep Q-Network

The network separates:

- State value estimation
- Action advantage estimation

This allows the model to better understand which states are valuable independently of specific actions.

### Soft Target Network Updates

A separate target network is updated gradually to improve training stability and prevent oscillations.

---

## Tech Stack

- Python
- PyTorch
- NumPy
- CUDA
- Deep Learning
- Reinforcement Learning

---

## Hardware Acceleration

The training process supports GPU acceleration using CUDA, allowing neural network training to be significantly faster compared to CPU-only execution.

---

## Project Structure

Example structure:
