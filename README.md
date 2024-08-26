# Banana-Collector-environment with Double DQN

This project applies a Deep Dueling Double Q-Network (DDQN) to train an agent in the Unity Banana Collector environment

## Table of Contents
- [Installation](#installation)
- [File Descriptions](#File-Descriptions)
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

1. **Banana_Collector_agent.ipynb**: Notebook contains the implementation and training procedure for the Banana Collector agent using DDQN
2. **ddqn_online_model.pth** This file contains the saved weights of the online network used in the Deep Q-Network (DDQN) implementation.
3. **ddqn_target_model.pth** This file contains the saved weights of the target network in the DDQN framework.
4. **Report.md** This file contains The project report





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
