# Banana-Collector-environment with Double DQN

This project implements a Deep Dueling Double Q-Network (DDQN) using the Unity Banana Collector environment.

## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Environment Setup](#environment-setup)
- [Running the Model](#running-the-model)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Acknowledgments](#acknowledgments)

## Installation

```python
pip install unityagents
pip install torch
pip install numpy
pip install matplotlib
```
## File Descriptions

1. **-**: Notebook contains the data analysis
2. **ddqn_online_model.pth** This file contains the saved weights of the online network used in the Deep Q-Network (DDQN) implementation.
3. **ddqn_target_model.pth** This file contains the saved weights of the target network in the DDQN framework.




## Saving and Loading the Model

### Saving the Model

After training your DDQN model, you can save the model's parameters using the following command:

```python
torch.save(online_network.state_dict(), 'ddqn_online_model.pth')
torch.save(target_network.state_dict(), 'ddqn_target_model.pth')
```

### Loading the Model
```python
online_network = QNetwork(state_size, action_size).to(device)
target_network = QNetwork(state_size, action_size).to(device)

online_network.load_state_dict(torch.load('ddqn_online_model.pth'))
target_network.load_state_dict(torch.load('ddqn_target_model.pth'))
```

## Acknowledgements
This project uses the Unity ML-Agents Toolkit and the Banana Collector environment provided by [Udacity](https://www.udacity.com/).
