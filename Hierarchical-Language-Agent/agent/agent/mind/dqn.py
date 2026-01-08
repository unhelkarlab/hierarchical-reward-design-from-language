import gym

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

import matplotlib.pyplot as plt

import random


def linear_schedule_for_epsilon(initial_value, final_value, duration, timestep):
  """Implements a linear scheduler for epsilon.
    
    Args:
        initial_value: (float) Initial value of epsilon.
        final_value: (float) Final value of epsilon.
        duration: (int) Duration over which to decay epsilon from its initial to final value.
        timestep: (int) The current time step.
    
    Returns:
        Value of epsilon at the given timestep.
    """
  slope = (final_value - initial_value) / duration
  return max(slope * timestep + initial_value, final_value)


class QNetwork(nn.Module):
  """Approximates the Q Function as a Multi-Layer Perceptron."""

  def __init__(self, env, nodes_per_mlp_layer=[32, 32]):
    """Initialize the Q Function apprixmated as a Multi-Layer Perceptron.
        
        Args:
            env: An OpenAI Gym environment.
            nodes_per_mlp_layer: An array of integers. The length of array equals the number
                of hidden layers of the Multi-Layer Perceptron. Each element in the array
                equals the number of nodes in the corresponding layer.
        """
    super().__init__()

    ######## PUT YOUR CODE HERE ########
    # input layer
    self.input_layer = nn.Linear(env.observation_space.shape[0],
                                 nodes_per_mlp_layer[0])

    # hidden layers
    self.hidden_layers = nn.ModuleList()
    self.hidden_layers.append(nn.ReLU())
    for k in range(len(nodes_per_mlp_layer) - 1):
      self.hidden_layers.append(
          nn.Linear(nodes_per_mlp_layer[k], nodes_per_mlp_layer[k + 1]))
      self.hidden_layers.append(nn.ReLU())

    # output layer
    self.output_layer = nn.Linear(nodes_per_mlp_layer[-1], env.action_space.n)

    self.q_network = nn.Sequential(self.input_layer, *self.hidden_layers,
                                   self.output_layer)
    ######## PUT YOUR CODE HERE ########

  def forward(self, x):
    """Implements the forward pass of the Q Network.
        
        Args:
            x: Input to the Q Network.
        
        Returns:
            Output of the Q Network.
        """
    ######## PUT YOUR CODE HERE ########
    return self.q_network(x)
    ######## PUT YOUR CODE HERE ########


class ReplayBuffer:
  """A buffer to store agent's experiences."""

  def __init__(self, env, buffer_size):
    """Initialize a ring buffer to store agent's experiences.
        
        Args:
            env: An OpenAI Gym environment.
            buffer_size: An integer. The total size of the buffer.
        """
    observation_n = env.observation_space.shape[0]
    self.buffer_size = buffer_size

    self.observations = np.zeros((self.buffer_size, observation_n),
                                 dtype=np.float32)
    self.next_observations = np.zeros((self.buffer_size, observation_n),
                                      dtype=np.float32)
    self.actions = np.zeros((self.buffer_size, ), dtype=np.int64)
    self.rewards = np.zeros((self.buffer_size, ), dtype=np.float32)
    self.dones = np.zeros((self.buffer_size, ), dtype=np.float32)

    ######## PUT YOUR CODE HERE ########
    ######## PUT YOUR CODE HERE ########

  def add(self, state, action, next_state, reward, done):
    """Add an experience to the buffer.
        
        Args:
            state: the current environment state
            action: the action executed in the state
            next_state: the state after executing the action
            reward: the reward received after executing the action
            done: Boolean denoting whether the task is completed.        
        """
    ######## PUT YOUR CODE HERE ########
    self.observations = np.roll(self.observations, -1, axis=0)
    self.observations[-1] = state

    self.next_observations = np.roll(self.next_observations, -1, axis=0)
    self.next_observations[-1] = next_state

    self.actions = np.roll(self.actions, -1)
    self.actions[-1] = action

    self.rewards = np.roll(self.rewards, -1)
    self.rewards[-1] = reward

    self.dones = np.roll(self.dones, -1)
    self.dones[-1] = done
    ######## PUT YOUR CODE HERE ########

  def sample(self, batch_size):
    """Sample a mini-batch of experiences from the replay buffer.
        
        Args:
            batch_size: An integer. The size of the mini-batch.
        
        Returns:
            Randomly sampled experiences from the replay buffer.
        """
    indices = np.random.randint(self.buffer_size, size=batch_size)
    observations = torch.from_numpy(self.observations[indices])
    next_observations = torch.from_numpy(self.next_observations[indices])
    actions = torch.from_numpy(self.actions[indices])
    rewards = torch.from_numpy(self.rewards[indices])
    dones = torch.from_numpy(self.dones[indices])

    return observations, actions, next_observations, rewards, dones


class AgentBase:

  def __init__(self, env):
    self.env = env
    self.num_actions = self.env.action_space.n
    self.policy = self.make_policy()
    self.behavior_policy = self.make_behavior_policy()

  def make_policy(self):
    """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and return action
        """
    raise NotImplementedError

  def make_behavior_policy(self):
    """
        Similar to make_policy, it returns a policy function. But this one used
        for interaction with the environment.
        """
    raise NotImplementedError

  def run_episode(self, episode_policy):
    """
        Generate one episode with the given policy
        """
    episode = []
    done = False
    obs = self.env.reset()
    episode_return = 0
    while not done:
      action = episode_policy(obs)
      next_obs, reward, done, _ = self.env.step(action)
      episode.append([obs, action, reward, next_obs, done])
      obs = next_obs
      episode_return += reward

    return (episode, episode_return)

  def evaluate(self, num_eval_episodes=1000, plot_title="Evaluation"):
    """Evaluates the agent."""
    list_returns = []
    list_average_returns = []
    average_return = 0
    for episode_idx in range(num_eval_episodes):
      _, episode_return = self.run_episode(self.policy)
      average_return += (1. /
                         (episode_idx + 1)) * (episode_return - average_return)
      list_returns.append(episode_return)
      list_average_returns.append(average_return)

    print(f"Average reward {round(average_return, 3)}")
    plt.plot(list_returns, '^', label="Return")
    plt.plot(list_average_returns, 'r', label="Average Return")
    plt.ylabel('Return')
    plt.xlabel('Episode#')
    plt.title(plot_title)
    plt.legend()
    plt.ylim(-501, 0.0)
    plt.show()


class DeepQLearning(AgentBase):
  """Implements a Q Learner with function approximation."""

  def __init__(self,
               env,
               buffer_size=200,
               batch_size=8,
               initial_epsilon=1.0,
               final_epsilon=0.01,
               epsilon_decay_duration=1000,
               learning_rate=0.001,
               num_gradient_updates=1,
               q_network_update_frequency=1,
               target_network_update_frequency=5,
               learning_starts_at_step=10):
    """Initializes the Agent.
        
        Args:
            env: An OpenAI Gym environment.
            buffer_size: (integer) Size of the replay buffer.
            batch_size: (integer) Size of the mini batch.
            initial_epsilon: (float) Initial value of epsilon.
            final_epsilon: (float) Final value of epsilon.
            epsilon_decay_duration: (integer) Duration over which to decay epsilon.
            learning_rate: (float) Learning rate for Q network update.
            num_gradient_updates: (integer) Number of stochastic gradient updates with each minibatch.
            q_network_update_frequency: (integer) Steps after which to update Q network.
            target_network_update_frequency: (integer) Steps after which to update target network.            
            learning_starts_at_step: (integer) Step at which to begin learning. Before this, the 
                agent explores and collects experiences in its replay buffer.
        """
    super().__init__(env=env)

    self.gamma = 0.99  # Assume a discount factor of 0.99
    self.current_step = 0
    self.learning_starts_at_step = learning_starts_at_step
    self.batch_size = batch_size
    self.num_gradient_updates = num_gradient_updates
    self.q_network_update_frequency = q_network_update_frequency
    self.target_network_update_frequency = target_network_update_frequency

    # Create exploration scheduler
    self.epsilon_scheduler = lambda current_step: linear_schedule_for_epsilon(
        initial_epsilon, final_epsilon, epsilon_decay_duration, current_step)

    ######## PUT YOUR CODE HERE ########
    self.replay_buffer = ReplayBuffer(env, buffer_size)
    self.q_net = QNetwork(env)
    self.target_q_net = QNetwork(env)
    self.target_q_net.load_state_dict(self.q_net.state_dict())
    self.opt = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
    self.steps_since_update = 0
    ######## PUT YOUR CODE HERE ########

    self.state = env.reset()
    self.list_returns = []
    self.list_average_returns = []
    self.average_return = 0.
    self.list_moving_average_returns = []
    self.moving_average_returns_by_step = np.empty([200000])

  @property
  def epsilon(self):
    return self.epsilon_scheduler(self.current_step)

  def make_policy(self):

    def policy_func(observation):
      ######## PUT YOUR CODE HERE ########
      return self.q_net(torch.Tensor(observation)).max(-1)[1].item()
      ######## PUT YOUR CODE HERE ########

    return policy_func

  def make_behavior_policy(self):

    def policy_func(observation):
      ######## PUT YOUR CODE HERE ########
      if random.random() < self.epsilon:
        action = self.env.action_space.sample()
      else:
        action = self.q_net(torch.Tensor(observation)).max(-1)[1].item()
      return action
      ######## PUT YOUR CODE HERE ########

    return policy_func

  def train_step(self, states, actions, rewards, next_states, mask,
                 target_network, num_actions):
    with torch.no_grad():
      # Taking the max along the last axis for (batch_size, action_space_size)
      # So q_vals_ns has shape (batch_size,)
      q_vals_ns = target_network(next_states).max(-1)[0]

    self.opt.zero_grad()
    # q_vals will have shape (batch_size, action_space_size)
    q_vals = self.q_net(states)
    # one_hot_actions has shape (batch_size, action_space_size)
    one_hot_actions = torch.nn.functional.one_hot(torch.LongTensor(actions),
                                                  num_actions)

    loss = ((rewards + self.gamma * mask * q_vals_ns -
             torch.sum(q_vals * one_hot_actions, -1))**2).mean()
    loss.backward()
    self.opt.step()

    return loss

  def update(self):
    """Update the agent."""
    ######## PUT YOUR CODE HERE ########
    # Only start learning after a certain number of steps.
    if self.current_step <= self.learning_starts_at_step:
      return

    print('update q')
    self.steps_since_update += 1

    # Sample states, actions, next_states, rewards, and dones from the replay buffer.
    states, actions, next_states, rewards, dones = self.replay_buffer.sample(
        self.batch_size)
    mask = torch.stack(
        [torch.Tensor([0]) if done else torch.Tensor([1]) for done in dones])

    # Call helper function to update the agent's q network.
    self.train_step(states, actions, rewards, next_states, mask,
                    self.target_q_net, self.env.action_space.n)

    # Update the target network according to the update frequency.
    if self.steps_since_update + 1 == self.target_network_update_frequency:
      self.target_q_net.load_state_dict(self.q_net.state_dict())
      self.steps_since_update = 0
      print('update target')
    ######## PUT YOUR CODE HERE ########

  def train(self, num_train_episodes, make_plot=False, break_out=False):

    for episode_idx in range(num_train_episodes):
      # Reset environment before beginning the episode
      done = False
      self.state = self.env.reset()
      episode_return = 0

      # Run the episode and update the policy
      while not done:
        # First, generate a step with behavior policy
        action = self.behavior_policy(self.state)
        next_state, reward, done, _ = self.env.step(action)

        # Update the replay buffer
        self.replay_buffer.add(self.state, action, next_state, reward, done)

        # Second, update the agent
        self.update()

        # Prepare for next step
        self.state = next_state
        self.current_step += 1
        episode_return += reward
        if self.current_step < 200000:
          if len(self.list_moving_average_returns) > 0:
            self.moving_average_returns_by_step[
                self.current_step] = self.list_moving_average_returns[-1]
          else:
            self.moving_average_returns_by_step[self.current_step] = -500.

        if self.current_step % 10000 == 0:
          print(
              f"Timestep: {self.current_step}, episode reward (moving average, 20 episodes): {round(self.list_moving_average_returns[-1],2)}"
          )

      # Store the return for evaluation
      self.list_returns.append(episode_return)
      self.average_return = np.mean(np.asarray(self.list_returns))
      self.list_average_returns.append(self.average_return)

      if len(self.list_returns) > 20:
        self.list_moving_average_returns.append(
            np.mean(np.asarray(self.list_returns[-20:])))
      else:
        self.list_moving_average_returns.append(self.average_return)

      if break_out:
        if self.current_step >= 100000:
          break

    if make_plot:
      plt.plot(self.list_returns, '^', label="Return")
      plt.plot(self.list_average_returns,
               'r',
               label="Average Return (all episodes)")
      plt.plot(self.list_moving_average_returns,
               'b',
               label="Average Return (last 20 episodes)")
      plt.ylabel('Return')
      plt.xlabel('Episode#')
      plt.title('Performance during training')
      plt.ylim(-501, 0.0)
      plt.legend()
      plt.show()
