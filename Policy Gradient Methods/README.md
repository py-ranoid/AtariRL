# Playing Atari Games with Reinforcement Learning 🤖 

This project aims to implement Reinforcement Learning (RL) algorithms for playing OpenAI's gym. We will be using OpenAI's Gym library to provide the environment and PyTorch for implementing the RL algorithms.

Prerequisites
To run this project, you need to have Python 3.9 or later installed. You also need to install the following dependencies:
- OpenAI Gym
- PyTorch
- Matplotlib

You can install the dependencies using the following commands:
```
pip install gymnasium
pip install matplotlib
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```

## Usage
To train the policy gradient algorithms using a specific policy algorithm, run the following command:
```
python train.py --policy <policy_algorithm>
```
where <policy_algorithm> can be one of the following:
- policy_gradient: Vanilla Policy Gradient
- actor_critic: Actor Critic
- acer: Actor Critic with Experience Replay
- vtrace: V-Trace
- ppo: Proximal Policy Optimization

The training rewards can be downloaded and unzipped from 'rewards.zip'