import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


def get_onehot(a, l=5):
    x = torch.zeros(l)
    x[a] = 1
    return x


# 定义 ICM 模块
class ICMModule(nn.Module):
    """
    内在好奇心模块，用于计算好奇心损失
    """
    def __init__(self, input_dim, action_dim):
        super(ICMModule, self).__init__()  # 调用父类的构造函数
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),  # 定义特征提取网络
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(128 + action_dim, 128),  # 定义前向预测网络
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(256, 128),  # 定义反向预测网络
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, next_state, action):
        """
        前向传播

        参数：
        state (Tensor)：当前状态
        next_state (Tensor)：下一个状态
        action (Tensor)：采取的动作

        返回：
        Tuple[Tensor, Tensor]：前向损失和反向损失
        """
        state_feature = self.feature(state)  # 提取当前状态的特征
        next_state_feature = self.feature(next_state)  # 提取下一个状态的特征
        predicted_next_state_feature = self.forward_model(torch.cat((state_feature, action), dim=1))  # 预测下一个状态的特征
        predicted_action = self.inverse_model(torch.cat((state_feature, next_state_feature), dim=1))  # 预测采取的动作
        forward_loss = nn.MSELoss()(predicted_next_state_feature, next_state_feature)  # 计算前向损失
        inverse_loss = nn.CrossEntropyLoss()(predicted_action, action.argmax(dim=1))  # 计算反向损失
        return forward_loss + inverse_loss, forward_loss, inverse_loss  # 返回总损失、前向损失、反向损失


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
                 lmbda, eps, gamma, device, icm_lr=1e-3, curiosity_beta=0.01):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.icm = ICMModule(state_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.icm_optimizer = optim.Adam(list(self.icm.parameters()), lr=icm_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.curiosity_beta = curiosity_beta  # 好奇心损失的权重

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, curiosity_loss_list, forward_loss_list, inverse_loss_list):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        actions_ = torch.tensor(transition_dict['actions_']).view(-1, self.action_dim).to(self.device)

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
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        for _ in range(10):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            curiosity_loss, forward_loss, inverse_loss = self.icm(states, next_states, actions_.float())  # 计算好奇心损失
            curiosity_loss_list.append(curiosity_loss.detach().numpy())
            forward_loss_list.append(forward_loss.detach().numpy())
            inverse_loss_list.append(inverse_loss.detach().numpy())
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            self.icm_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            curiosity_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.icm_optimizer.step()

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
