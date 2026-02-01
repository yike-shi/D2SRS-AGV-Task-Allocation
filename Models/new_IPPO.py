import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

import sys
target_path='./'
sys.path.append(target_path)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.sigmoid(self.fc2(F.sigmoid(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc2(F.sigmoid(self.fc1(x))))
        return self.fc3(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps,
                            1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save_model(self, actor_path='.\\Net\\Actor\\', critic_path='.\\Net\\Critic\\'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path='.\\Net\\Actor\\', critic_path='.\\Net\\Critic\\'):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
    
    def load_model_params(self, actor_params, critic_params):
        """
        加载策略网络和价值网络的参数
        :param actor_params: 要加载到策略网络的参数
        :param critic_params: 要加载到价值网络的参数
        # """
        # # 遍历网络参数并打印它们所在的设备
        # for name, param in self.actor.named_parameters():
        #     print(f"{name}: {param.device}")
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)
        # # 遍历网络参数并打印它们所在的设备
        # for name, param in self.actor.named_parameters():
        #     print(f"{name}: {param.device}")


class PartialResetPPO(PPO):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        super().__init__(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                        lmbda, eps, gamma, device)
        
        # 保存初始参数（需要放在构造函数最后）
        self.initial_actor_params = copy.deepcopy(self.actor.state_dict())
        self.initial_critic_params = copy.deepcopy(self.critic.state_dict())
         # 添加部分重置方法：重置Critic网络参数
        self.critic_init_state = copy.deepcopy(self.critic.state_dict())  # 保存初始参数

    def partial_reset(self, layer_patterns=['fc']):
        """部分重置Critic网络：只重置最后两层参数"""
        # 获取初始化的参数
        init_state = self.critic_init_state
        current_state = self.critic.state_dict()
        # 保留Actor相关参数，只重置Critic的最后两层
        for key in current_state:
            if 'critic.fc3' in key or 'critic.fc4' in key: 
                current_state[key] = init_state[key]
        self.critic.load_state_dict(current_state)
        print("Critic网络部分重置完成")