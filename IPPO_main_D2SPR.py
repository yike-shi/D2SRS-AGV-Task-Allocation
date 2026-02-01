from copy import deepcopy
import json
import sys
import time
import pandas as pd
import csv
import os
import warnings
warnings.filterwarnings("ignore")

from Envs.ENV import WorkShopEnv
target_path='./'
sys.path.append(target_path)


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from log.logging_setting import getLogging
from Models.rl_utils import *
from Models.new_IPPO import PPO
from Models.new_IPPO import PartialResetPPO
from collections import deque

def save_train_data(dict, s, a, n_s, r, d):
    dict['states'].append(s)
    dict['actions'].append(a)
    dict['next_states'].append(n_s)
    dict['rewards'].append(r)
    dict['dones'].append(d)

def write_lists_to_csv(lists, names, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入列表名称到第一行    
        csv_writer.writerow(['Index'] + names)
        max_length = max(len(lst) for lst in lists)
        for i in range(max_length):
            row = [i + 1]  # 添加行号
            for lst in lists:
                if i < len(lst):
                    row.append(lst[i])
                else:
                    row.append('')
            csv_writer.writerow(row)

def save_fig(figname, data):
    plt.plot(data)
    plt.title(figname)
    plt.savefig(f'{figname}.png')
    plt.close()

# 用于计算变化率的队列
select_returns_queue = deque(maxlen=10)
agv_returns_queue = deque(maxlen=10)

def calculate_change_rate(queue):
    if len(queue) < 2:
        return float('inf')  # 如果队列长度小于2，则认为变化率为无穷大
    if queue[0] == 0:
        return float('inf')  
    return abs((queue[-1] - queue[0]) / queue[0])


if __name__ == '__main__':
    
    logger.info(f'当前订单生成模式：自动随机生成')
    order_nums = int(input('输入自动生成订单数量: ')) + 1
    actor_lr = 3e-4
    critic_lr = 1e-3
    #num_episodes = 1000 + 20
    num_episodes = 500
    max_steps = 50000
    hidden_dim = 128
    gamma = 0.99
    lmbda = 0.97
    eps = 0.2
    convergence_threshold = 1e-3  # 收敛阈值
    # device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    
    logger = getLogging()
    env = WorkShopEnv(logger=logger, time_needed=False)
    select_state_dim = env.select_observation_space
    select_action_dim = env.select_action_space
    agv_state_dim = env.agv_observation_space
    agv_action_dim = env.agv_action_space

    # 构造智能体
    select_agent = PartialResetPPO(select_state_dim, hidden_dim, select_action_dim, 
                                actor_lr, critic_lr, lmbda, eps, gamma, device)
    agv_agent = PartialResetPPO(agv_state_dim, hidden_dim, agv_action_dim,
                            actor_lr, critic_lr, lmbda, eps, gamma, device)

    #修改return_list
    random.seed(1)
    np.random.seed(1)

    # 加载网络参数
    #select_agent.load_model(actor_path=f'.//Net//select_agent_net//Actor//episode200.pth', critic_path=f'.//Net//select_agent_net//Critic//episode200.pth')
    #agv_agent.load_model(actor_path=f'.//Net//agv_agent_net//Actor//episode200.pth', critic_path=f'.//Net//agv_agent_net//Critic//episode200.pth')

    agv_return_list = []
    select_return_list = []
    finish_goods_list = []
    order_intime_list = []
    select_penalty = []
    task_assign_wrong_penalty = []
    task_to_errly_penalty = []
    task_to_late_penalty = []
    task_finish_reward = []
    finish_goods_reward = []
    total_task_num = []
    total_calculate_num = []
    total_task_mean_time = []
    total_task_correct = []
    total_task_early = []
    total_task_late = []

    for episode in range(num_episodes):
            state, reward, done = env.reset()
            env.control_center.change_order_num(order_nums)
            #select_state, agv1_state, agv2_state, agv3_state,agv4_state,agv5_state = state
            select_state, agv1_state, agv2_state, agv3_state= state
            episode_return = 0
            select_episode_return = 0
            agv_episode_return = 0
            order_in_time = 0
            
            # 构造数据集，保存每个回合的状态数据
            select_transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }

            agv_transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }

            cost_time = 0

            for step in range(max_steps):
                # select_agent
                start_time = time.time()
                select_a = select_agent.take_action(state=select_state)

                agv1_a = agv_agent.take_action(state=agv1_state)
                agv2_a = agv_agent.take_action(state=agv2_state)    
                agv3_a = agv_agent.take_action(state=agv3_state)
                #agv4_a = agv_agent.take_action(state=agv4_state)
                #agv5_a = agv_agent.take_action(state=agv5_state)

                end_time = time.time()

                #if is_count:
                cost_time += end_time - start_time
                    
                #actions = select_a, agv1_a, agv2_a, agv3_a, agv4_a, agv5_a
                actions = select_a, agv1_a, agv2_a, agv3_a
                next_state, reward, done = env.step(step, actions)
                #next_select_state, next_agv1_state, next_agv2_state, next_agv3_state ,next_agv4_state ,next_agv5_state = next_state
                next_select_state, next_agv1_state, next_agv2_state, next_agv3_state = next_state
                order_in_time = env.control_center.process_order()
                # 保存每个时刻的状态\动作\..
                save_train_data(select_transition_dict, select_state, select_a, next_select_state, reward[1], done[0])
                save_train_data(agv_transition_dict, agv1_state, agv1_a, next_agv1_state, reward[0], done[0])
                save_train_data(agv_transition_dict, agv2_state, agv2_a, next_agv2_state, reward[0], done[0])
                save_train_data(agv_transition_dict, agv3_state, agv3_a, next_agv3_state, reward[0], done[0])
                #save_train_data(agv_transition_dict, agv4_state, agv4_a, next_agv4_state, reward[0], done[0])
                #save_train_data(agv_transition_dict, agv5_state, agv5_a, next_agv5_state, reward[0], done[0])
                
                # 更新状态
                select_state = next_select_state
                agv1_state = next_agv1_state
                agv2_state = next_agv2_state
                agv3_state = next_agv3_state
                #agv4_state = next_agv4_state
                #agv5_state = next_agv5_state
                
                # 更新奖励
                agv_episode_return +=reward[0]
                select_episode_return +=reward[1]

                if done[0]:
                    logger.warning(f'订单完成时间：{step}')
                    break


            logger.info(f'生成方案时间：{cost_time:.4f}s')
            
            order_json = {}
            order_json["orders"] = []
            for i in range(2, order_nums+1):
                _order = {}
                order = env.control_center.order_manager.get_order_by_id(i)
                _order["name"] = f'order{order["order_ID"]-1}'
                _order["time"] = {}
                _order["time"]["earlist"] = order["earlist"]
                _order["time"]["latest"] = order["lastest"]
                _order["target"] = {}
                _order["target"]["product1"] = order["products"]
                _order["agvIdOfTask"] = []
                task_id_list = []
                task_id_list.append(order["task1_id"])
                task_id_list.append(order["task2_id"])
                task_id_list.append(order["task3_id"])
                task_id_list.append(order["task4_id"])
                task_id_list.append(order["task5_id"])

                for task_id in task_id_list:
                    task = env.control_center.task_manager.get_task_by_id(task_id)
                    #_order["agvIdOfTask"].append(task["AGV_id"])
                order_json['orders'].append(_order)


            # 将字典保存为 JSON 文件
            with open('order.json', 'w') as json_file:
                json.dump(order_json, json_file, indent=4)
            logger.info("JSON 文件已保存")
            #  保存每个回合的return
            agv_return_list.append(agv_episode_return)
            select_return_list.append(select_episode_return)
            finish_goods_list.append(env.control_center.finishgoods_storage.remain_goods)
            #
            order_intime_list.append(order_in_time)
            
            total_task_num.append(env.control_center.get_task_total_nums)
            total_calculate_num.append(env.control_center.total_calculate_num)
            total_task_correct.append(env.control_center.total_task_correct)
            total_task_early.append(env.control_center.total_task_early)
            total_task_late.append(env.control_center.total_task_late)
            total_task_mean_time.append(env.control_center.total_task_mean_time)

            # 模型训练
            select_agent.update(select_transition_dict)
            agv_agent.update(agv_transition_dict)
            logger.warning(f'生产总产品数量{env.control_center.finishgoods_storage.remain_goods}个')
            logger.warning(f'任务平均执行时间{env.control_center.total_task_mean_time}')
            logger.warning(f'============= END =============')

            # 计算变化率并决定是否切换奖励模式
            select_returns_queue.append(select_episode_return)
            agv_returns_queue.append(agv_episode_return)

            select_change_rate = calculate_change_rate(select_returns_queue)
            agv_change_rate = calculate_change_rate(agv_returns_queue)

            if (select_change_rate < convergence_threshold and agv_change_rate < convergence_threshold) or episode == 251:
                env.sparse_mode = True  # 启用稀疏奖励
                logger.info("========切换到稀疏奖励模式========")

            # 保存网络
            if (episode+1) % 50 == 0:
                logger.info("process data")
                # 构建路径
                select_actor_path = os.path.join('.', 'Net', 'select_agent_net', 'Actor', f'episode{episode + 1}.pth')
                select_critic_path = os.path.join('.', 'Net', 'select_agent_net', 'Critic', f'episode{episode + 1}.pth')
                agv_actor_path = os.path.join('.', 'Net', 'agv_agent_net', 'Actor', f'episode{episode + 1}.pth')
                agv_critic_path = os.path.join('.', 'Net', 'agv_agent_net', 'Critic', f'episode{episode + 1}.pth')

                select_agent.save_model(actor_path=select_actor_path, critic_path=select_critic_path)
                agv_agent.save_model(actor_path=agv_actor_path, critic_path=agv_critic_path)

                if env.sparse_mode == False:
                    save_fig('select_return_list_d', select_return_list)
                    save_fig('agv_return_list_d', agv_return_list)
                    save_fig('finish_goods_list_d', finish_goods_list)
                    save_fig('order_intime_list_d', order_intime_list)
                    save_fig('total_task_mean_time_d', total_task_mean_time)            
                    names = ['select_return_list','agv_return_list', 'finish_goods_list','order_intime_list', 'total_task_mean_time' ]
                    lists = [select_return_list, agv_return_list, finish_goods_list, order_intime_list, total_task_mean_time]
                    output_dir = os.path.join('.', 'results')
                    os.makedirs(output_dir, exist_ok=True)
                    write_lists_to_csv(lists, names, os.path.join(output_dir, 'data_d.csv'))
                    logger.info(f"data.csv 已保存到 {output_dir}")
                else:
                    save_fig('select_return_list_s', select_return_list)
                    save_fig('agv_return_list1_s', agv_return_list)
                    save_fig('finish_goods_list_s', finish_goods_list)
                    save_fig('order_intime_list', order_intime_list)
                    save_fig('total_task_mean_time', total_task_mean_time)            
                    names = ['select_return_list','agv_return_list', 'finish_goods_list','order_intime_list', 'total_task_mean_time' ]
                    lists = [select_return_list, agv_return_list, finish_goods_list, order_intime_list, total_task_mean_time]
                    output_dir = os.path.join('.', 'results')
                    os.makedirs(output_dir, exist_ok=True)
                    write_lists_to_csv(lists, names, os.path.join(output_dir, 'data.csv'))
                    logger.info(f"data.csv 已保存到 {output_dir}")

   