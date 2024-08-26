# Report on Deep Q-Network (DQN) Implementation

## 1. Introduction

This report details the implementation of a Deep Q-Network (DQN) used for reinforcement learning.

## 2. Learning Algorithm

The algorithm implemented is a variant of the DQN known as Double DQN (DDQN), which helps to mitigate the overestimation bias inherent in traditional DQNs by using two separate networks: an online network and a target network.

### 2.1 Hyperparameters

- **Number of Episodes (`n_episodes`)**: Determines the number of episodes the agent will interact with the environment.
- **Batch Size (`batch_size`)**: Size of the mini-batch sampled from the replay buffer for each learning step. Default value: 64.
- **Discount Factor (`gamma`)**: Factor used to discount future rewards. Default value: 0.98.
- **Learning Rate (`lr`)**: Learning rate for the optimizer. Default value: 5e-4.
- **Epsilon Start (`start`)**: Initial value for the epsilon parameter in the epsilon-greedy policy. Default value: 0.8.
- **Epsilon End (`end`)**: Final value for epsilon. Default value: 0.05.
- **Epsilon Decay (`decay`)**: Number of steps over which epsilon is decayed. Default value: 800.
- **Soft Update Factor (`tau`)**: Factor for updating the target network with a soft update. Default value: 0.005.

## 3. Model Architectures

### 3.1 QNetwork

The `QNetwork` class defines the architecture of the neural network used for approximating the Q-values. The network consists of:

- **Input Layer**: Takes state information as input.
- **Hidden Layers**:
  - **First Hidden Layer**: Fully connected layer with `fc1_units` units (default: 64) and ReLU activation.
  - **Second Hidden Layer**: Fully connected layer with `fc2_units` units (default: 64) and ReLU activation.
- **Output Layer**: Fully connected layer that outputs Q-values for each action.

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        # Instantiate the first hidden layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # Instantiate the output layer
        self.fc3 = nn.Linear(fc2_units, action_size)
    def forward(self, state):
        # Ensure the ReLU activation function is used
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

### 3.2 Target and Online Networks

The `create_target_network_and_online_network` function initializes both the target network and the online network, ensuring they start with the same weights.

```python
def create_target_network_and_online_network(state_size, action_size, device):
    online_network = QNetwork(state_size, action_size).to(device)
    target_network = QNetwork(state_size, action_size).to(device)
    target_network.load_state_dict(online_network.state_dict())
    return target_network, online_network
```

### 3.3 Replay Buffer

The `ReplayBuffer` class handles experience replay by storing and sampling experience tuples. This helps to break the temporal correlation between consecutive experiences and stabilizes training.

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.memory)
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
```
## 4. Training Procedure

The `DDQN` function orchestrates the training process. It initializes the optimizer, performs training over a specified number of episodes, and updates both the online and target networks.

```python
def DDQN(n_episodes, batch_size=64, gamma=0.98, lr=5e-4, start=.8, end=.05, decay=800):
    optimizer = optim.Adam(online_network.parameters(), lr=lr)
    total_steps = 0
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.from_numpy(env_info.vector_observations[0]).float().to(device)
        episode_reward = 0
        while True:
            total_steps += 1
            q_values = online_network(state)
            action = select_action(q_values, total_steps, start, end, decay)
            env_info = env.step(action)[brain_name]
            next_state = torch.from_numpy(env_info.vector_observations[0]).float().to(device)
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                q_values = online_network(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_actions = online_network(next_states).argmax(1).unsqueeze(1)
                    next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_target_network(target_network, online_network, tau=.005)
            
            state = next_state
            episode_reward += reward
            if done:
                break

        scores_window.append(episode_reward)
        scores.append(episode_reward)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tepisode_reward: {episode_reward}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tepisode_reward: {episode_reward}')
        
        if np.mean(scores_window) >= 13:
            break
            
    torch.save(online_network.state_dict(), 'ddqn_online_model.pth')
    torch.save(target_network.state_dict(), 'ddqn_target_model.pth')
    return scores
```

## 5. Idea for Future Work

### Prioritized Experience Replay

To enhance the learning efficiency, consider implementing **Prioritized Experience Replay (PER)**. Unlike uniform sampling, PER prioritizes experiences with higher temporal-difference (TD) errors, which are more informative. This approach can accelerate learning and improve policy performance.

