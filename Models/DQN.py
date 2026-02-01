# Standard library imports
import random
from collections import deque

# Third-party library imports
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.Sigmoid(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class DQNAgent:

    def __init__(self, state_dim, action_dim, use_conv=True, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(self.state_dim, self.action_dim).to(self.device)
            self.target_model = ConvDQN(self.state_dim, self.action_dim).to(self.device)
        else:
            self.model = DQN(self.state_dim, self.action_dim).to(self.device)
            self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if (np.random.randn() < eps):
            return np.random.randint(0, self.action_dim)

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def save_model(self, path='.\\Models\\'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='.\\Models\\'):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):

    with tqdm(total=max_episodes, desc='Processing') as pbar:

        episode_rewards = []

        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward

                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)
                
                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    # print("Episode " + str(episode) + ": " + str(episode_reward))
                    pbar.set_postfix({"return": episode_reward})
                    pbar.update(1)
                    break

                state = next_state

    return episode_rewards


if __name__ == '__main__':
    env_id = "CartPole-v1"
    MAX_EPISODES = 500
    MAX_STEPS = 200
    BATCH_SIZE = 32

    RETURN_LIST = []  # 保存每回合的reward
    LOSS_LIST = []  # 保存每一step的动作价值函数


    env = gym.make(env_id)
    state = env.reset()
    agent = DQNAgent(env, use_conv=False)
    RETURN_LIST = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

    plt.plot(RETURN_LIST)
    plt.title('return')
    plt.savefig('result.png')


