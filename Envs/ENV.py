import random
import sys

from gym import Space
import numpy as np
import torch

from log.logging_setting import getLogging
from Entity.entity_file import *

def time_to_category(time_length, max_time=10000, interval=5000):
    """
    将相对时间长度转换为对应的类别索引。
    
    参数:
    - time_length: 需要转换的时间长度。
    - max_time: 最大时间长度，默认为50000。
    - interval: 每个区间的大小，默认为1000。
    
    返回:
    - category_index: 对应的时间长度所属类别的索引。
    """
    # 确定总共有多少个类别
    num_categories = max_time // interval
    
    # 如果时间长度超出最大时间，则将其归类到最后一个类别
    if time_length >= max_time:
        return num_categories - 1
    
    # 计算所属的类别索引
    category_index = (time_length - 1) // interval
    
    return category_index

def time_to_category200(time_length, max_time=30000, interval=3000):
    """
    将相对时间长度转换为对应的类别索引。
    
    参数:
    - time_length: 需要转换的时间长度。
    - max_time: 最大时间长度，默认为20000。
    - interval: 每个区间的大小，默认为1000。
    
    返回:
    - category_index: 对应的时间长度所属类别的索引。
    """
    # 确定总共有多少个类别
    num_categories = max_time // interval
    
    # 如果时间长度超出最大时间，则将其归类到最后一个类别
    if time_length >= max_time:
        return num_categories - 1
    
    # 计算所属的类别索引
    category_index = ((time_length - 1) // interval)+1
    if time_length ==0:
        category_index = 0
    
    return category_index

class SelectHeuristic:
    def __init__(self, agv1, agv2, agv3=None, agv4=None, agv5=None, agv_num=3):
        if agv_num == 2:
            self.agvs = [agv1, agv2]
        elif agv_num == 3:
            self.agvs = [agv1, agv2, agv3]
        elif agv_num == 5:
            self.agvs = [agv1, agv2, agv3, agv4, agv5]
    
    # def forward(self):
    #     min_num = 10
    #     min_id = 0
    #     for agv in self.agvs:
    #         if not agv.accept_task:
    #             continue
    #         if len(agv.todo_list) < min_num:
    #             min_num = len(agv.todo_list)
    #             min_id = agv.id
    #     return min_id

    def forward(self):
        id = random.randint(0, len(self.agvs))
        return id 



class AGVHeuristic:
    def __init__(self, algorithm, task_manager) -> None:
        algorithm_dic = {'FCFS':1, 'SJF':2, 'EDD':3, 'HPR':4, 'RAND':5, 'LCFS':6}
        self.algorithm = algorithm_dic.get(algorithm)  # 1 先入先出FIFO， 2 先入后出LIFO 3 最短作业路径优先SJF 4 先完成先服务FCFS 5 随机Rand
        self.task_manager = task_manager

    def forward(self, lst, agv):
        if self.algorithm == 1:
            num, lst = self.FCFS(lst)
        elif self.algorithm == 2:
            num, lst = self.SJF(lst, agv)
        elif self.algorithm == 3:
            num, lst = self.EDD(lst)
        elif self.algorithm == 4:
            num, lst = self.HPR(lst)
        elif self.algorithm == 5:
            num, lst = self.Rand(lst)
        elif self.algorithm == 6:
            num, lst = self.LCFS(lst)
        return num, lst

    def FCFS(self, lst):
        num = lst[0] if lst else None  # 根据FCFS逻辑定义num
        if num is not None:
            lst.remove(num)  # 从列表中删除已选择的元素
        return num, lst
    
    def HPR(self, lst):  # 根据任务优先级选择任务
        max_pro = 0
        max_id = None
        for taskid in lst:
            task = self.task_manager.get_task_by_id(taskid)
            category = task['category']
            if 1 <= category and category <= 3:
                if max_pro < 1:
                    max_id = taskid
                    max_pro = 1
            if 4 <= category and category <= 9:
                if max_pro < 2:
                    max_id = taskid
                    max_pro = 2
            if 10 <= category and category <= 11:
                if max_pro < 3:
                    max_id = taskid
                    max_pro = 3
            if 12 <= category and category <= 13:
                if max_pro < 4:
                    max_id = taskid
                    max_pro = 4
        if max_id is not None:
            lst.remove(max_id)
        return max_id, lst
    
    def SJF(self, lst, agv):
        ax = agv.position_x
        ay = agv.position_y
        min_dis = 1e6
        min_id = None
        task = None
        for taskid in lst:
            task = self.task_manager.get_task_by_id(taskid)
            tx1, ty1 = task['starting point']
            tx2, ty2 = task['ending point']
            dis1 = L1(ax, ay, tx1, ty1)
            dis2 = L1(tx1, ty1, tx2, ty2)
            dis = dis1 + dis2
            if dis < min_dis:
                min_dis = dis
                min_id = taskid
        if min_id is not None:
            lst.remove(min_id)
        return min_id, lst

    def EDD(self, lst):  # 最早截止时间优先
        latest_time = 1e6
        latest_id = None
        for taskid in lst:
            task = self.task_manager.get_task_by_id(taskid)
            if task['latest time'] < latest_time:
                latest_time = task['latest time']
                latest_id = taskid
        if latest_id is not None:
            lst.remove(latest_id)
        return latest_id, lst
    
    def Rand(self, lst):
        num = random.choice(lst) if lst else None  # 根据Rand逻辑定义num
        if num is not None:
            lst.remove(num)  # 从列表中删除已选择的元素
        return num, lst
    
    def LCFS(self, lst):
        num = lst[-1] if lst else None  # 根据FCFS逻辑定义num
        if num is not None:
            lst.remove(num)  # 从列表中删除已选择的元素
        return num, lst


class WorkShopEnv:
    def __init__(self, logger=None, MADDPG=False, agv_num=3, time_needed=False) -> None:
        self.agv_num = agv_num
        self.__select_action_space = 1 + self.agv_num
        #self.__select_observation_space = 4 + 8 * self.agv_num #将AGV待处理任务数量上限改为2个，原是8 * self.agv_num
        self.time_needed = time_needed       
        self.__agv_action_space = 6
        if self.time_needed:
            self.__agv_observation_space = 52 #将AGV待处理任务数量上限修改为2个，原来为52
            self.__select_observation_space = 4 + 8 * self.agv_num  # 将AGV待处理任务数量上限改为2个，原是8 * self.agv_num
        else:
            self.__agv_observation_space = 42 # 减去AGV可接任务数（当前为5）✖️2
            self.__select_observation_space = 2 + 8 * self.agv_num  # 将AGV待处理任务数量上限改为2个，原是8 * self.agv_num
        self.__reward_space = 1
        self.logger = logger
        self.entity_manager = None
        self.task_manager = None
        self.order_manager = None
        self.agv1 = None
        self.agv2 = None
        self.agv3 = None
        self.agv4 = None
        self.agv5 = None
        self.agvs = [self.agv1, self.agv2, self.agv3, self.agv4, self.agv5]
        self.assigning_task = None
        self.time = 0
        self.agents = 4
        self.MADDPG = MADDPG
        self.time_needed = time_needed
        # 新增奖励模式标志位
        self.sparse_mode = False  

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        # 生成车间实体对象
        global ENTITY_ID
        global TASK_ID
        ENTITY_ID = 0
        TASK_ID = 0
        self.time = 0 
        self.entity_manager = EntityManager()
        self.task_manager = TaskManager()
        self.order_manager = OrderManager()
        ENTITY_ID += 1
        matrial_storage = MaterialStorage(GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, logger=self.logger)
        ENTITY_ID += 1
        self.control_center = ControlCenter(GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, task_manager=self.task_manager, order_manager=self.order_manager, logger=self.logger)
        ENTITY_ID += 1
        production_line_1 = ProductionLine(in_x=300, in_y=800, out_x=700, out_y=800, target_num=1, id=1,
                                        GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, logger=self.logger, in_volume_limit=800, control_center=self.control_center)
        ENTITY_ID += 1
        production_line_2 = ProductionLine(in_x=300, in_y=600, out_x=700, out_y=600, target_num=2, id=2,
                                        GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, logger=self.logger, control_center=self.control_center)
        ENTITY_ID += 1
        production_line_3 = ProductionLine(in_x=300, in_y=400, out_x=700, out_y=400, target_num=3, id=3,
                                        GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, logger=self.logger, control_center=self.control_center)
        ENTITY_ID += 1
        finish_goods_storage = FinishGoodsStorage(GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager,
                                                pos_x=900, pos_y=100, logger=self.logger, control_center=self.control_center)
        ENTITY_ID += 1
        assembly_stand1 = AssemblyStand(in_x=800, in_y=700, out_x=900, out_y=700, id=1, GLOBAL_ID=ENTITY_ID, 
                                        entity_manager=self.entity_manager, logger=self.logger, control_center=self.control_center)
        ENTITY_ID += 1
        assembly_stand2 = AssemblyStand(in_x=800, in_y=500, out_x=900, out_y=500, id=2, GLOBAL_ID=ENTITY_ID, 
                                        entity_manager=self.entity_manager, logger=self.logger, control_center=self.control_center)
        ENTITY_ID += 1
        tools_storage = ToolStorage(GLOBAL_ID=ENTITY_ID, entity_manager=self.entity_manager, logger=self.logger, control_center=self.control_center)
        self.task_manager.get_control_center(self.control_center)
        
        # AGV setting
        agv_range_limit = 20000
        agv_speed = 10

        ENTITY_ID += 1
        self.agv1 = AGV(range_limit=agv_range_limit, volume_limit=2000, weight_limit=20000, speed=agv_speed, id=1, GLOBAL_ID=ENTITY_ID, task_manager=self.task_manager, control_center=self.control_center, logger=self.logger)
        ENTITY_ID += 1
        self.agv2 = AGV(range_limit=agv_range_limit, volume_limit=2000, weight_limit=20000, speed=agv_speed, id=2, GLOBAL_ID=ENTITY_ID, task_manager=self.task_manager, control_center=self.control_center, logger=self.logger)
        # self.agv2.remain_range = 10000
        ENTITY_ID += 1
        self.agv3 = AGV(range_limit=agv_range_limit, volume_limit=2000, weight_limit=20000, speed=agv_speed, id=3, GLOBAL_ID=ENTITY_ID, task_manager=self.task_manager, control_center=self.control_center, logger=self.logger)
        # self.agv3.remain_range = 5000
        ENTITY_ID += 1
        self.agv4 = AGV(range_limit=agv_range_limit, volume_limit=2000, weight_limit=20000, speed=agv_speed, id=3, GLOBAL_ID=ENTITY_ID, task_manager=self.task_manager, control_center=self.control_center, logger=self.logger)
        ENTITY_ID += 1
        self.agv5 = AGV(range_limit=agv_range_limit, volume_limit=2000, weight_limit=20000, speed=agv_speed, id=3, GLOBAL_ID=ENTITY_ID, task_manager=self.task_manager, control_center=self.control_center, logger=self.logger)
                
        self.entity_manager.add_entity(matrial_storage)
        self.entity_manager.add_entity(production_line_1)
        self.entity_manager.add_entity(production_line_2)
        self.entity_manager.add_entity(production_line_3)
        self.entity_manager.add_entity(assembly_stand1)
        self.entity_manager.add_entity(assembly_stand2)
        self.entity_manager.add_entity(finish_goods_storage)
        self.entity_manager.add_entity(tools_storage)
        self.entity_manager.add_entity(self.agv1)
        self.entity_manager.add_entity(self.agv2)
        self.entity_manager.add_entity(self.agv3)
        self.entity_manager.add_entity(self.agv4)
        self.entity_manager.add_entity(self.agv5)
                
        # 控制中心读取各个车间实体对象
        self.control_center.get_info(Material_GLOBLA_ID=matrial_storage.GLOBAL_ID,
                            Produce_Line1_ID=production_line_1.GLOBAL_ID,
                            Produce_Line2_ID=production_line_2.GLOBAL_ID,
                            Produce_Line3_ID=production_line_3.GLOBAL_ID,
                            Assembly_stand1_ID=assembly_stand1.GLOBAL_ID,
                            Assembly_stand2_ID=assembly_stand2.GLOBAL_ID,
                            FinishGoods_ID=finish_goods_storage.GLOBAL_ID,
                            ToolsStorage_ID=tools_storage.GLOBAL_ID,
                            AGV1_GLOBAL_ID=self.agv1.GLOBAL_ID,
                            AGV2_GLOBAL_ID=self.agv2.GLOBAL_ID,
                            AGV3_GLOBAL_ID=self.agv3.GLOBAL_ID,
                            AGV4_GLOBAL_ID=self.agv4.GLOBAL_ID,
                            AGV5_GLOBAL_ID=self.agv5.GLOBAL_ID,
                            time_needed=self.time_needed)
        self.agvs = [self.agv1, self.agv2, self.agv3, self.agv4, self.agv5]

        self.control_center.reset_task()
        self.control_center.reset_order()
        
        self.control_center.generate_orders(0, 1000, 2)
        self.control_center.process_order()
        #self.control_center.generate_task(1)
        #self.control_center.generate_task(2)
        #self.control_center.generate_task(3)

        task = self.find_suitable_task()
        if task:
            select_state = self.update_select_state(task, time=0)
        
        agv1_state = [-1] * self.agv_observation_space
        agv2_state = [-1] * self.agv_observation_space
        agv3_state = [-1] * self.agv_observation_space
        agv4_state = [-1] * self.agv_observation_space
        agv5_state = [-1] * self.agv_observation_space

        reward_select = 0
        reward_agv = 0
        done = False
        if self.agv_num < 3:
            state = [select_state, agv1_state, agv2_state]
        elif self.agv_num == 3:
            state = [select_state, agv1_state, agv2_state, agv3_state]
        elif self.agv_num > 3:
            state = [select_state, agv1_state, agv2_state, agv3_state, agv4_state, agv5_state]

        state = np.array(state, dtype=object)

        reward = [reward_select, reward_agv]
        done = [done, done, done, done]

        return state, reward, done


    def find_suitable_task(self):
        # 找到最合适的任务
        if self.control_center.unassigned_task:
            task = self.task_manager.get_task_by_id(self.control_center.unassigned_task[0])
            self.assigning_task = task
            return task
        else:
            self.assigning_task = None
            return False
        

    def step(self, time, actions):
        """ select agent 根据 unassigning_task 分配任务 """
        """ agv agent 在agv的todo_list大于2的时候对执行顺序进行安排 """
        """ 目前未完成奖励设置：根据任务完成时间是否在要求的时间内确定"""
        """ 如果agv完成某个任务的时间早于 最早完成时间，则agv等待，给惩罚"""
        #print(time)
        if (time+1) % 3000 == 0:
        #     agv1_use = self.agv1.working_time / 50000
        #     agv2_use = self.agv1.working_time / 50000
        #     agv3_use = self.agv1.working_time / 50000
        #     self.control_center.select_reward += agv1_use * 5000 + agv2_use * 5000 + agv3_use * 5000
            #finish_goods = self.control_center.finishgoods_storage.remain_goods
            #self.control_center.select_reward += finish_goods
            #self.control_center.agv_reward += finish_goods
            #print(self.control_center.order_intime)
            self.control_center.select_reward += self.control_center.order_intime*1000
            self.control_center.agv_reward += self.control_center.order_intime*1000
        #if time % 399 == 0:
        #     agv1_use = self.agv1.working_time / 50000
        #     agv2_use = self.agv1.working_time / 50000
        #     agv3_use = self.agv1.working_time / 50000
        #     self.control_center.select_reward += agv1_use * 5000 + agv2_use * 5000 + agv3_use * 5000
        #    finish_goods = self.control_center.finishgoods_storage.remain_goods
        #    self.control_center.select_reward += finish_goods*10.0
        #    self.control_center.agv_reward += finish_goods*10.0
        #     # correct = self.control_center.total_task_correct
        #     # total = self.control_center.get_task_total_nums
        #     # self.control_center.select_reward += (correct / total) * 10000.0
        #     # self.control_center.agv_reward += (correct / total) * 10000.0
        #     mean_time = self.control_center.total_task_mean_time
        #     self.control_center.select_reward += (1500-mean_time)/1500.0 * 10000.0
        #     self.control_center.agv_reward += (1500-mean_time)/1500.0 * 10000.0
        #     done = True


        if self.MADDPG:
            def get_action(actions):
                action_list = []
                for act in actions:
                    act = np.where(act == 1)
                    act = int(act[0])
                    action_list.append(act)
                return action_list
            actions = get_action(actions)
        
        agv1_action = 0 
        agv2_action = 0 
        agv3_action = 0 
        agv4_action = 0 
        agv5_action = 0
                       
        if self.agv_num < 3:
            select_action, agv1_action, agv2_action = actions
        elif self.agv_num == 3:
            select_action, agv1_action, agv2_action, agv3_action = actions
        elif self.agv_num > 3:
            select_action, agv1_action, agv2_action, agv3_action, agv4_action, agv5_action = actions
        
        reward_select = 0
        reward_agv = 0
        done = False
        select_state = None
        agv1_state = None
        agv2_state = None
        agv3_state = None
        agv4_state = None
        agv5_state = None

        if select_action != 0 and self.assigning_task == None:
            select_action = 0
            # self.control_center.reward -= 1.0

        if select_action == 0:
            # 如果一直不接任务会对效率有较大影响
            if self.assigning_task in self.control_center.unassigned_task:  # 将该任务放置最后选择
                self.control_center.unassigned_task.remove(self.assigning_task)
                self.control_center.unassigned_task.append(self.assigning_task)

            if (len(self.agv1.todo_list) + len(self.agv2.todo_list) + len(self.agv3.todo_list)) < 3:
                if self.agv1.accept_task or self.agv2.accept_task or self.agv3.accept_task:
                    self.control_center.select_reward -= 1.0

            if (len(self.agv1.todo_list) + len(self.agv2.todo_list) + len(self.agv3.todo_list)) > 12:
                self.control_center.select_reward += 1
            # 将AGV待处理任务上限修改为2个
       
        elif select_action == 1:
            self.control_center.task_allocation(self.agv1.GLOBAL_ID, self.assigning_task['task_ID'])
        elif select_action == 2:
            self.control_center.task_allocation(self.agv2.GLOBAL_ID, self.assigning_task['task_ID'])
        elif select_action == 3:
            self.control_center.task_allocation(self.agv3.GLOBAL_ID, self.assigning_task['task_ID'])
        elif select_action == 4:
            self.control_center.task_allocation(self.agv4.GLOBAL_ID, self.assigning_task['task_ID'])
        elif select_action == 5:
            self.control_center.task_allocation(self.agv5.GLOBAL_ID, self.assigning_task['task_ID'])

        if agv1_action != 0:
            if self.agv1.study_mark == True and agv1_action <= len(self.agv1.todo_list):
                self.control_center.agv_reward += 1.0
                self.agv1.study_mark = False
                task_id = self.agv1.todo_list[agv1_action - 1]
                self.agv1.todo_list.remove(task_id)
                self.agv1.doing_task_id = task_id
                # self.logger.warning(f'TIME={self.time}: AGV{self.agv1.id} 开始执行任务{task_id}')
                # self.agv1.todo_list.insert(0, task_id)
            # else:
            #     self.control_center.reward -= 1.0
                # self.control_center.select_penalty -= 1.0
        else:
            if self.agv1.study_mark == True and len(self.agv1.todo_list)>0:
                self.control_center.agv_reward -= 10.0

        if agv2_action != 0:
            if self.agv2.study_mark == True and agv2_action <= len(self.agv2.todo_list):
                self.control_center.agv_reward += 1.0
                self.agv2.study_mark = False
                task_id = self.agv2.todo_list[agv2_action - 1]
                self.agv2.todo_list.remove(task_id)
                self.agv2.doing_task_id = task_id
                # self.agv2.todo_list.insert(0, task_id)
            # else:
            #     self.control_center.reward -=1.0
                # self.control_center.select_penalty -=1.0
        else:
            if self.agv2.study_mark == True and len(self.agv2.todo_list) > 0:
                self.control_center.agv_reward -= 10.0
        
        if self.agv_num == 3:
            if agv3_action != 0:
                if self.agv3.study_mark == True and agv3_action <= len(self.agv3.todo_list):
                    self.control_center.agv_reward += 1.0
                    self.agv3.study_mark = False
                    task_id = self.agv3.todo_list[agv3_action - 1]
                    self.agv3.todo_list.remove(task_id)
                    self.agv3.doing_task_id = task_id
                    # self.agv3.todo_list.insert(0, task_id)
                # else:
                #     self.control_center.reward -=1.0
                    # self.control_center.select_penalty -=1.0
            else:
                if self.agv3.study_mark == True and len(self.agv3.todo_list)>0:
                    self.control_center.agv_reward -= 10.0

        if self.agv_num == 5:
            if agv4_action != 0:
                if self.agv4.study_mark == True and agv4_action <= len(self.agv4.todo_list):
                    self.control_center.agv_reward += 1.0
                    self.agv4.study_mark = False
                    task_id = self.agv4.todo_list[agv4_action - 1]
                    self.agv4.todo_list.remove(task_id)
                    self.agv4.doing_task_id = task_id
                    # self.agv4.todo_list.insert(0, task_id)
                # else:
                #     self.control_center.reward -=1.0
                    # self.control_center.select_penalty -=1.0
            else:
                if self.agv4.study_mark == True and len(self.agv4.todo_list)>0:
                    self.control_center.agv_reward -= 10.0
            
            if agv5_action != 0:
                if self.agv5.study_mark == True and agv5_action <= len(self.agv5.todo_list):
                    self.control_center.agv_reward += 1.0
                    self.agv5.study_mark = False
                    task_id = self.agv5.todo_list[agv5_action - 1]
                    self.agv5.todo_list.remove(task_id)
                    self.agv5.doing_task_id = task_id
                    # self.agv5.todo_list.insert(0, task_id)
                # else:
                #     self.control_center.reward -= 1.0
                    # self.control_center.select_penalty -=1.0
            else:
                if self.agv5.study_mark == True and len(self.agv5.todo_list)>0:
                    self.control_center.agv_reward -= 10.0
        
        #is_count = 0
        # select_agent
        task = self.find_suitable_task()
        if task:
            select_state = self.update_select_state(task, time)
            #is_count += 1
        else:
            select_state = [-1] * self.select_observation_space

        # agv agent
        agv = self.agv1
        if agv.study_mark and agv.state == 0 and len(agv.todo_list) > 0:
            # agv.study_mark = False
            # 对 agv1 的待处理人物的执行顺序进行安排
            agv1_state = self.update_agv_state(time=time, agv=agv)
        else:
            agv1_state = [-1] * self.agv_observation_space

        agv = self.agv2
        if agv.study_mark and agv.state == 0  and len(agv.todo_list) > 0:
            # agv.study_mark = False
            # 对 agv2 的待处理人物的执行顺序进行安排
            agv2_state = self.update_agv_state(time=time, agv=agv)
        else:
            agv2_state = [-1] * self.agv_observation_space
        
        agv = self.agv3
        if agv.study_mark and agv.state == 0 and len(agv.todo_list) > 0:
            # agv.study_mark = False
            # 对 agv2 的待处理人物的执行顺序进行安排
            agv3_state = self.update_agv_state(time=time, agv=agv)
        else:
            agv3_state = [-1] * self.agv_observation_space
        
        agv = self.agv4
        if agv.study_mark and agv.state == 0 and len(agv.todo_list) > 0:
            # agv.study_mark = False
            # 对 agv2 的待处理人物的执行顺序进行安排
            agv4_state = self.update_agv_state(time=time, agv=agv)
        else:
            agv4_state = [-1] * self.agv_observation_space
        
        agv = self.agv5
        if agv.study_mark and agv.state == 0 and len(agv.todo_list) > 0:
            # agv.study_mark = False
            # 对 agv2 的待处理人物的执行顺序进行安排
            agv5_state = self.update_agv_state(time=time, agv=agv)
        else:
            agv5_state = [-1] * self.agv_observation_space
            
        self.control_center.step(time=time)

        # reward
        reward_agv = self.control_center.agv_reward
        reward_select = self.control_center.select_reward

        # 稀疏奖励模式覆盖
        if self.sparse_mode: 
            reward_agv = 0
            reward_select = 0
            if (self.control_center.order_intime+1) % 10 ==0: 
                reward_agv = self.control_center.order_intime *1000
                reward_select = self.control_center.order_intime *1000

        
        reward = [reward_agv,reward_select]
        self.control_center.agv_reward = 0
        self.control_center.select_reward = 0
        
        done = self.control_center.done

        
        done = [done, done, done, done]

        

        if self.agv_num == 2:
            next_state = (select_state, agv1_state, agv2_state)
        elif self.agv_num == 3:
            next_state = (select_state, agv1_state, agv2_state, agv3_state)
        else:
            next_state = (select_state, agv1_state, agv2_state, agv3_state, agv4_state, agv5_state)

                
        return next_state, reward, done
    

    def update_select_state(self, task, time):
        """
            select agent 根据当前任务、AGV的状态选择具体执行任务的agv
            状态：当前时间，待处理任务任务类型，待处理任务最早送达时间，待处理任务最晚送达时间  4
            AGV1 是否接受任务，剩余续航，待处理任务数目，任务1类别，任务2类别，任务3类别，任务4类别, 任务5类别  8
            AGV2 是否接受任务，剩余续航，待处理任务数目，任务1类别，任务2类别，任务3类别，任务4类别, 任务5类别  8
            AGV3 是否接受任务，剩余续航，待处理任务数目，任务1类别，任务2类别，任务3类别，任务4类别, 任务5类别  8
            AGV4 是否接受任务，剩余续航，待处理任务数目，任务1类别，任务2类别，任务3类别，任务4类别, 任务5类别  8
            AGV5 是否接受任务，剩余续航，待处理任务数目，任务1类别，任务2类别，任务3类别，任务4类别, 任务5类别  8
        """
        state1 = []
        state1.append(time)
        state1.append(task['category'])
        if self.time_needed:
            state1.append(task['earliest time'])
            state1.append(task['latest time'])
        def build_agv_state(agv):
            state = []
            if agv.accept_task:
                state.append(1)
            else:
                state.append(0)
            state.append(time_to_category200(agv.remain_range))
            state.append(len(agv.todo_list))
            #state.append(action)  # 添加当前动作到状态
        
            todo_num = 0
            for i in range(len(agv.todo_list)):
                if i == 5:
                    break  # 可能出现第六个任务，第6个是充电任务
                task_id = agv.todo_list[i]
                task = self.control_center.task_manager.get_task_by_id(task_id)
                try:
                    state.append(task['category'])
                except:
                    state.append(0)
                todo_num += 1
            for _ in range(todo_num, 5):
                state.append(0)

            return state
        state2 = build_agv_state(self.agv1)
        state3 = build_agv_state(self.agv2)
        if self.agv_num == 3:
            state4 = build_agv_state(self.agv3)
        elif self.agv_num > 3:
            state4 = build_agv_state(self.agv3)
            state5 = build_agv_state(self.agv4)
            state6 = build_agv_state(self.agv5)

        if self.agv_num < 3:
            state = state1 + state2 + state3
        elif self.agv_num == 3:
            state = state1 + state2 + state3 + state4
        elif self.agv_num > 3:
            state = state1 + state2 + state3 + state4 + state5 + state6

        return state
        '''state2 = []
        agv = self.agv1
        if agv.accept_task:
            state2.append(1)
        else:
            state2.append(0)
        state2.append(agv.remain_range)
        state2.append(len(agv.todo_list))
        todo_num = 0
        for i in range(len(agv.todo_list)):
            if i == 5:
                break  # 可能出现第六个任务，第6个是充电任务
            task_id = agv.todo_list[i]
            task = self.control_center.task_manager.get_task_by_id(task_id)
            try:
                state2.append(task['category'])
            except:
                state2.append(0)
            todo_num += 1
        for _ in range(todo_num, 5):
            state2.append(0)

        state3 = []
        agv = self.agv2
        if agv.accept_task:
            state3.append(1)
        else:
            state3.append(0)
        state3.append(agv.remain_range)
        state3.append(len(agv.todo_list))
        todo_num = 0
        for i in range(len(agv.todo_list)):
            if i == 5:
                break  # 可能出现第六个任务，第6个是充电任务
            task_id = agv.todo_list[i]
            task = self.control_center.task_manager.get_task_by_id(task_id)
            try:
                state3.append(task['category'])
            except:
                state3.append(0)
            todo_num += 1
        for _ in range(todo_num, 5):
            state3.append(0)
        
        state4 = []
        agv = self.agv3
        if agv.accept_task:
            state4.append(1)
        else:
            state4.append(0)
        state4.append(agv.remain_range)
        state4.append(len(agv.todo_list))
        todo_num = 0
        for i in range(len(agv.todo_list)):
            if i == 5:
                break  # 可能出现第六个任务，第6个是充电任务
            task_id = agv.todo_list[i]
            task = self.control_center.task_manager.get_task_by_id(task_id)
            try:
                state4.append(task['category'])
            except:
                state4.append(0)
            todo_num += 1
        for _ in range(todo_num, 5):
            state4.append(0)

        state5 = []
        agv = self.agv4
        if agv.accept_task:
            state5.append(1)
        else:
            state5.append(0)
        state5.append(agv.remain_range)
        state5.append(len(agv.todo_list))
        todo_num = 0
        for i in range(len(agv.todo_list)):
            if i == 5:
                break  # 可能出现第六个任务，第6个是充电任务
            task_id = agv.todo_list[i]
            task = self.control_center.task_manager.get_task_by_id(task_id)
            try:
                state5.append(task['category'])
            except:
                state5.append(0)
            todo_num += 1
        for _ in range(todo_num, 5):
            state5.append(0)
        
        state6 = []
        agv = self.agv5
        if agv.accept_task:
            state6.append(1)
        else:
            state6.append(0)
        state6.append(agv.remain_range)
        state6.append(len(agv.todo_list))
        todo_num = 0
        for i in range(len(agv.todo_list)):
            if i == 5:
                break  # 可能出现第六个任务，第6个是充电任务
            task_id = agv.todo_list[i]
            task = self.control_center.task_manager.get_task_by_id(task_id)
            try:
                state6.append(task['category'])
            except:
                state6.append(0)
            todo_num += 1
        for _ in range(todo_num, 5):
            state6.append(0)

        if self.agv_num < 3:
            state = state1 + state2 + state3
        elif self.agv_num == 3:
            state = state1 + state2 + state3 + state4
        elif self.agv_num > 3:
            state = state1 + state2 + state3 + state4 + state5 + state6

        return state'''
    

    def update_agv_state(self, time, agv):
        '''
        更新状态
        状态：当前时间、生产线1原材料储备、生产线2原材料储备、生产线3原材料储备、装配台1原材料储备、装配台2原材料储备、装配台1工具储备、装配台2工具储备、  12
        任务1类型、任务最早送达时间、任务最晚送达时间、起点距当前位置距离、起点距离Todo-list[1]距离、起点距离Todo-list[2]距离、起点距离Todo-list[3]距离、起点距离Todo-list[4]距离  8
        任务2类型、任务最早送达时间、任务最晚送达时间、起点距当前位置距离、起点距离Todo-list[0]终点距离、起点距离Todo-list[2]距离、起点距离Todo-list[3]距离、起点距离Todo-list[4]距离
        任务3类型、任务最早送达时间、任务最晚送达时间、起点距当前位置距离、起点距离Todo-list[0]终点距离、起点距离Todo-list[1]距离、起点距离Todo-list[3]距离、起点距离Todo-list[4]距离
        任务4类型、任务最早送达时间、任务最晚送达时间、起点距当前位置距离、起点距离Todo-list[0]终点距离、起点距离Todo-list[1]距离、起点距离Todo-list[2]距离、起点距离Todo-list[4]距离
        任务5类型、任务最早送达时间、任务最晚送达时间、起点距当前位置距离、起点距离Todo-list[0]终点距离、起点距离Todo-list[1]距离、起点距离Todo-list[2]距离、起点距离Todo-list[3]距离
        '''

        state1 = []
        state1.append(time)
        state1.append(self.control_center.pLine1_raw_materials)
        state1.append(self.control_center.pLine2_raw_materials)
        state1.append(self.control_center.pLine3_raw_materials)
        state1.append(self.control_center.assembly1_raw_materials[0])
        state1.append(self.control_center.assembly1_raw_materials[1])
        state1.append(self.control_center.assembly1_raw_materials[2])
        state1.append(self.control_center.assembly_stand1.tools_num)
        state1.append(self.control_center.assembly2_raw_materials[0])
        state1.append(self.control_center.assembly2_raw_materials[1])
        state1.append(self.control_center.assembly2_raw_materials[2])
        state1.append(self.control_center.assembly_stand2.tools_num)

        if -1 in agv.todo_list:
            agv.todo_list.remove(-1)
            agv.todo_list.append(-1)

        for i in range(len(agv.todo_list)):
            task_id = agv.todo_list[i]
            if task_id == -1:
                break
            task = self.control_center.task_manager.get_task_by_id(task_id)
            
            state2 = []
            state2.append(task['category'])
            if self.time_needed:
                state2.append(task['earliest time'])
                state2.append(task['latest time'])

            task_x, task_y = task['starting point']
            state2.append(L1(agv.position_x, agv.position_y, task_x, task_y))
            todo_num = 0
            for j in range(len(agv.todo_list)):
                if i == j:
                    continue
                if j == 5:
                    break  # 可能有第六个任务为充电任务
                task_id = agv.todo_list[j]
                _task = self.control_center.task_manager.get_task_by_id(task_id)
                x, y = _task['ending point']
                state2.append(L1(x, y, task_x, task_y))
                todo_num += 1
            if self.time_needed:
                while len(state2)<8:
                    state2.append(-1.0)
            else:
                while len(state2)<6:
                    state2.append(-1.0)
            state1 = state1 + state2

        state3 = []
        for _ in range(5-len(agv.todo_list)):
            if self.time_needed:
                state3 = state3 + [-1.0] * 8
            else:
                state3 = state3 + [-1.0] * 6
        
        if -1 in agv.todo_list and len(agv.todo_list) <= 5:
            if self.time_needed:
                state3 = state3 + [-1.0] * 8
            else:
                state3 = state3 + [-1.0] * 6
        state1 = state1 + state3

        return state1

    @property
    def select_observation_space(self):
        return self.__select_observation_space

    @property
    def select_action_space(self):
        return self.__select_action_space

    @property
    def agv_observation_space(self):
        return self.__agv_observation_space

    @property
    def agv_action_space(self):
        return self.__agv_action_space
    
    @property
    def reward_space(self):
        return self.__reward_space

 
if __name__ == '__main__':
    logger = getLogging()
    logger.info('================= A New Begining =================')
    env = WorkShopEnv(logger=logger)
    state_ = env.reset()
    #heuristic = AGVHeuristic(algorithm='RAND', task_manager=env.task_manager)
    # select_heuristic = SelectHeuristic(env.agv1, env.agv2, env.agv3)
    # for i in range(13):
    #     task = env.control_center.generate_task(i+1)
    #     # print(f'i == {i}, task = {task}')
    # env.agv1.todo_list = [7, 8, 9, 10]
    # env.agv2.todo_list = [7, 8, 10]
    # env.agv3.todo_list = [7, 8, 9, 10]

    
    # print(heuristic.forward(env.agv1.todo_list, env.agv1))
    # print(select_heuristic.forward())


    # state = env.update_agv_state(10, env.agv1)
    # print(state)
    # print(len(state))

    # select_state, agv1_state, agv2_state, reward, done, select_study, agv1_study, agv2_study = env.step(10)
    # print(f'select_study = {select_study} select_state = {select_state}, len = {len(select_state)}')
    # print(f'agv1_study = {agv1_study} agv1_state = {agv1_state} len(agv1_state) = {len(agv1_state[0])}')
    # print(f'agv2_study = {agv2_study} agv2_state = {agv2_state}')

    # for state in agv1_state:
    #     print(state)
