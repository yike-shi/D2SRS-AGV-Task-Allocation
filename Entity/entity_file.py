import sys

import numpy as np

from log.logging_setting import getLogging

# global parameter
AGV_ORIGINAL_X = 100
AGV_ORIGINAL_Y = 700
METRIAL_A_VOLUME = 2
METRIAL_A_WEIGHT = 1
METRIAL_B_VOLUME = 2
METRIAL_B_WEIGHT = 2
METRIAL_C_VOLUME = 4
METRIAL_C_WEIGHT = 1
METRIAL_D_VOLUME = 3
METRIAL_D_WEIGHT = 2
SEMI_FINISHED_1_VOLUME = 10
SEMI_FINISHED_1_WEIGHT = 5
SEMI_FINISHED_2_VOLUME = 9
SEMI_FINISHED_2_WEIGHT = 6
SEMI_FINISHED_3_VOLUME = 10
SEMI_FINISHED_3_WEIGHT = 8
FINISHED_GOODS_VOLUME = 35
FINISHED_GOODS_WEIGHT = 24
AGV_CHARGING_TIME = 3000  # AGV充电时间 3000 2500
# AGV_CHARGING_TIME = 2200  # AGV充电时间 3000 2500 
AGV_LOAD_TIME = 10  # AGV装载时间
AGV_UNLOAD_TIME = 10  # AGV卸载时间
AGV_VOLUME_LIMIT = 200  # AGV体积限制



# 小规模
# PRO_WAIT_TIME = 1500  # 生产线查询等待时间
# ASS_WAIT_TIME = 1000  # 任务查询等待时间
# 中规模 
# PRO_WAIT_TIME = 1000  # 生产线查询等待时间
# ASS_WAIT_TIME = 600  # 任务查询等待时间
# 大规模
PRO_WAIT_TIME = 500  # 生产线查询等待时间
ASS_WAIT_TIME = 300  # 任务查询等待时间

TASK_ID = 0  # 任务id
ENTITY_ID = 0  # 实体id
ORDER_ID = 0 # 订单id
ORDER_IN_TIME = 0 #按时完成的任务
ORDER_GENERATION_INTERVAL = 800  # 订单生成间隔
MAX_ORDER_NUM = 10  # 最大订单数


# 半成品1： 需要A：2,B:1,C:1,D:0, 生产需要30t, 每隔10t加入产线一个
PRODUCTION_REQUIREMENT = {1:[2, 1, 1, 0, 40, 15], 2:[2, 1, 0, 1, 70, 35], 3:[2, 2, 0, 1, 80, 30]}  # 0421
ASSEMBLY_REQUIREMENT = [2, 1, 1, 150, 50]  # 成品装配需要 半成品1 2件，半成品2 1件， 半成品3 1件 装配时间 50t, 每隔10t开始1个新的装配 # 0404部分收敛数据保存

PRODUCTION_REQUIREMENT = {1:[2, 1, 1, 0, 1, 1], 2:[2, 1, 0, 1, 1, 1], 3:[2, 2, 0, 1, 1, 1]}  # 0421
ASSEMBLY_REQUIREMENT = [2, 1, 1, 1, 1]  # 成品装配需要 半成品1 2件，半成品2 1件， 半成品3 1件 装配时间 50t, 每隔10t开始1个新的装配 # 0404部分收敛数据保存

# TASK_E1_TIME = 300
# TASK_L1_TIME = 1500

# TASK_EARLY_TIME = 300
# TASK_LATE_TIME = 1500

TASK_E1_TIME = 0
TASK_L1_TIME = 10

TASK_EARLY_TIME = 200
TASK_LATE_TIME = 800

# def L2(x1, y1, x2, y2):
#     """返回两个坐标的L2范式距离"""
#     return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def L1(x1, y1, x2, y2):
    """返回两个坐标的L1范式(曼哈顿)距离"""
    return abs(x1 - x2) + abs(y1 - y2)


class EntityManager:
    def __init__(self):
        self.entities = {}

    def add_entity(self, entity):
        self.entities[entity.GLOBAL_ID] = entity
        self.entities[entity.NAME] = entity

    def get_entity_by_id(self, entity_id):
        return self.entities.get(entity_id, None)

    def get_entity_by_name(self, entity_name):
        return self.entities.get(entity_name, None)


class TaskManager:
    def __init__(self) -> None:
        self.tasks = {}
        self.generate_charge_task()

    def generate_charge_task(self):
        task = {}
        task['task_ID'] = -1
        task['priority'] = 5  # 0-5 0最优先
        task['task create time'] = 0
        task['earliest time'] = 0
        task['latest time'] = 0
        task['starting point'] = (AGV_ORIGINAL_X, AGV_ORIGINAL_Y)
        task['ending point'] = (AGV_ORIGINAL_X, AGV_ORIGINAL_Y)
        task['category'] = 0
        task['a'] = 0
        task['b'] = 0
        task['c'] = 0
        task['d'] = 0
        task['1'] = 0
        task['2'] = 0
        task['3'] = 0
        task['goods'] = 0
        task['tools'] = 0
        task['state'] = 0  # 0未开始分配，1已经分配， 2已完成
        task['AGV_id'] = -1  # 0暂未分配
        task['end time'] = 0
        self.add_task(task)

    def add_task(self, task):
        self.tasks[task['task_ID']] = task

    def get_task_by_id(self, task_id):
        return self.tasks.get(task_id, None)

    def create_task(self, task_ID, starting_point, ending_point, category, priority=5, task_create_time=0,
                    earliest_time=0, latest_time=0, a=0, b=0, c=0, d=0, f1=0, f2=0, f3=0, goods=0, tools=0):
        task = {}
        task['task_ID'] = task_ID
        task['priority'] = priority  # 0-5 0最优先
        task['task create time'] = task_create_time
        task['earliest time'] = earliest_time
        # latest_time = latest_time + np.random.randint(100, 500)  # 任务完成时间额外添加富裕时间
        task['latest time'] = latest_time
        task['starting point'] = starting_point
        task['ending point'] = ending_point
        task['category'] = category
        task['a'] = a
        task['b'] = b
        task['c'] = c
        task['d'] = d
        task['1'] = f1
        task['2'] = f2
        task['3'] = f3
        task['goods'] = goods
        task['tools'] = tools
        task['state'] = 0  # 0未开始分配，1已经分配， 2已完成
        task['AGV_id'] = -1  # 0暂未分配
        task['end time'] = 0
        self.add_task(task)
        return task

    def get_control_center(self, control_center):
        self.control_center = control_center
  

class OrderManager:
    def __init__(self):
        self.orders = {}

    def add_order(self, order):
        self.orders[order['order_ID']] = order

    def create_order(self, order_ID, earlist, lastest, products, category=1, task1=0, task2=0, task3=0, task4=0, task5=0):
        order = {}
        order['order_ID'] = order_ID
        order['earlist'] = earlist
        order['lastest'] = lastest
        order['products'] = products
        order['task1'] = task1  # AGV id
        order['task1_id'] = task1  # task id
        order['task2'] = task2
        order['task2_id'] = task1  # task id
        order['task3'] = task3
        order['task3_id'] = task1  # task id
        order['task4'] = task4
        order['task4_id'] = task1  # task id
        order['task5'] = task5
        order['task5_id'] = task1  # task id
        order['state'] = 0  # 0-5, start-end
        order['doing'] = False
        order['finishtime'] = 0 
        order['category'] = category
        order['task_id'] = []
        self.add_order(order)
        return order
    
    def get_order_by_id(self, order_id):
        return self.orders.get(order_id, None)


class ControlCenter:
    """控制中心"""
    def __init__(self, GLOBAL_ID, entity_manager, task_manager, order_manager, logger=None):
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = 'ControlCenter'
        self.entity_manager = entity_manager
        self.task_manager = task_manager
        self.order_manager = order_manager
        self.task_list = []
        self.logger = logger
        self.pLine1_nums = 0
        self.pLine2_nums = 0
        self.pLine3_nums = 0
        self.ass1_nums = 0
        self.ass2_nums = 0
        self.task_priority_nums = {'1':0, '2':0, '3':0, '4':0, '5':0,'6':0, '7':0, '8':0, '9':0, '10':0}
        self.unassigned_task = []
        self.wait_to_upgrade_task = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'10':[],'11':[]}  # key 是路线
        self.task_cost_classes_list = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[],'10':[],'11':[]}  # key 是路线
        self.entity_pos_in = {}  # key-Global_ID value position(x,y) 只有in
        self.entity_pos_out = {}  # key-Global_ID value position(x,y) 只有out
        self.time = 0
        #修改
        self.select_reward = 0
        self.agv_reward = 0
        
        self.continue_list = []
        self.pLine1_raw_materials = 0
        self.pLine2_raw_materials = 0
        self.pLine3_raw_materials = 0
        self.assembly1_raw_materials = [0, 0, 0]
        self.assembly2_raw_materials = [0, 0, 0]
        self.pLine1_times = 0
        self.pLine2_times = 0
        self.pLine3_times = 0
        self.pLine1_mark = 0
        self.pLine2_mark = 0
        self.pLine3_mark = 0
        self.pLine1_nums_used = 0  # 记录已经规划的数量
        self.pLine2_nums_used = 0
        self.pLine3_nums_used = 0
        self.ass1_nums_used = 0
        self.ass2_nums_used = 0
        # self.select_penalty = 0
        # self.task_assign_wrong_penalty = 0
        # self.task_to_errly_penalty = 0
        # self.task_to_late_penalty = 0
        # self.task_finish_reward = 0
        # self.finishgoods_reward = 0
        self.total_task_num = 0  # 任务总数量
        self.total_task_early = 0  # 任务过早完成
        self.total_task_late = 0  # 任务过完完成
        self.total_task_correct = 0  # 任务正确完成
        self.total_task_mean_time = 0  # 任务平均消耗时长
        self.total_calculate_num = 0  # 计算平均消耗时长的数目
        self.pLine1_raw_senting = 0  # 产线1正在送货中的原材料数量
        self.pLine2_raw_senting = 0  # 产线2正在送货中的原材料数量
        self.pLine3_raw_senting = 0  # 产线3正在送货中的原材料数量
        self.time_needed = False
        self.order_last_generation = 0
        self.order_doing_list = []
        self.done = False
        self.generate_time = -1 
        self.order_intime = 0
    
    def generate_orders(self, earlist, lastest, products):
        global ORDER_ID
        ORDER_ID += 1
        num = np.random.random()
        if num < 0.5:
            category = 0    
        else:
            category = 1
        self.order_manager.create_order(ORDER_ID, earlist, lastest, products, category)
        self.order_doing_list.append(ORDER_ID)
        self.logger.info(f'TIME={self.time}: 生成订单{ORDER_ID}, 最早{earlist}, 最晚{lastest}, 产品{products}.') 
        if ORDER_ID != 1:
            print(f'Time:{self.time:4} 到达订单{ORDER_ID - 1}: 交付时间{{{earlist:4},{lastest:4}}}, 产品{category+1} 数量{products}')
        return ORDER_ID
        
    def get_info(self, Material_GLOBLA_ID, Produce_Line1_ID, Produce_Line2_ID, Produce_Line3_ID, \
                 Assembly_stand1_ID, Assembly_stand2_ID, FinishGoods_ID, ToolsStorage_ID,
                 AGV1_GLOBAL_ID=None, AGV2_GLOBAL_ID=None, AGV3_GLOBAL_ID=None, AGV4_GLOBAL_ID=None, AGV5_GLOBAL_ID=None,
                 time_needed=False):
        self.material_storage = self.entity_manager.get_entity_by_id(Material_GLOBLA_ID)
        self.entity_pos_out[Material_GLOBLA_ID] = self.material_storage.out_pos
        self.produce_line1 = self.entity_manager.get_entity_by_id(Produce_Line1_ID)
        self.entity_pos_in[Produce_Line1_ID] = self.produce_line1.in_pos
        self.entity_pos_out[Produce_Line1_ID] = self.produce_line1.out_pos
        self.produce_line2 = self.entity_manager.get_entity_by_id(Produce_Line2_ID)
        self.entity_pos_in[Produce_Line2_ID] = self.produce_line2.in_pos
        self.entity_pos_out[Produce_Line2_ID] = self.produce_line2.out_pos
        self.produce_line3 = self.entity_manager.get_entity_by_id(Produce_Line3_ID)
        self.entity_pos_in[Produce_Line3_ID] = self.produce_line3.in_pos
        self.entity_pos_out[Produce_Line3_ID] = self.produce_line3.out_pos
        self.assembly_stand1 = self.entity_manager.get_entity_by_id(Assembly_stand1_ID)
        self.entity_pos_in[Assembly_stand1_ID] = self.assembly_stand1.in_pos
        self.entity_pos_out[Assembly_stand1_ID] = self.assembly_stand1.out_pos
        self.assembly_stand2 = self.entity_manager.get_entity_by_id(Assembly_stand2_ID)
        self.entity_pos_in[Assembly_stand2_ID] = self.assembly_stand2.in_pos
        self.entity_pos_out[Assembly_stand2_ID] = self.assembly_stand2.out_pos
        self.finishgoods_storage = self.entity_manager.get_entity_by_id(FinishGoods_ID)
        self.entity_pos_in[FinishGoods_ID] = self.finishgoods_storage.in_pos
        self.tools_storage = self.entity_manager.get_entity_by_id(ToolsStorage_ID)
        self.entity_pos_out[ToolsStorage_ID] = self.tools_storage.out_pos
        if AGV1_GLOBAL_ID:
            self.agv1 = self.entity_manager.get_entity_by_id(AGV1_GLOBAL_ID)
        if AGV2_GLOBAL_ID:
            self.agv2 = self.entity_manager.get_entity_by_id(AGV2_GLOBAL_ID)
        if AGV3_GLOBAL_ID:
            self.agv3 = self.entity_manager.get_entity_by_id(AGV3_GLOBAL_ID)
        if AGV4_GLOBAL_ID:
            self.agv4 = self.entity_manager.get_entity_by_id(AGV4_GLOBAL_ID)
        if AGV5_GLOBAL_ID:
            self.agv5 = self.entity_manager.get_entity_by_id(AGV5_GLOBAL_ID)
        self.time_needed = time_needed
  
    def agv_set_destination(self, AGV_GLOBAL_ID, end_x, end_y):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        agv.set_destination(end_x=end_x, end_y=end_y)
    
    def get_nums_from_task(self, task_id):
        task = self.task_manager.get_task_by_id(task_id)
        nums_list = {}
        nums_list['a'] = task['a']
        nums_list['b'] = task['b']
        nums_list['c'] = task['c']
        nums_list['d'] = task['d']
        nums_list['1'] = task['1']
        nums_list['2'] = task['2']
        nums_list['3'] = task['3']
        nums_list['goods'] = task['goods']
        nums_list['tools'] = task['tools']
        # print(f'num_list = {nums_list}')
        return nums_list

    def agv_unloading_setting(self, AGV_GLOBAL_ID, place_GLOBAL_ID):
        """place 包含 产线、装配台、成品存储仓"""
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        place = self.entity_manager.get_entity_by_id(place_GLOBAL_ID)
        task_id = agv.doing_task_id
        task = self.task_manager.get_task_by_id(task_id)
        
        earliest_time = task['earliest time']
        latest_time = task['latest time']

        # 每完成一次运输任务，奖励 100.0
        self.agv_reward += 100.0

        if self.time > earliest_time and self.time < latest_time:
            # self.task_finish_reward += 1.0
            if self.time_needed:
                self.agv_reward += 100.0
            self.total_task_correct += 1
            self.logger.info(f'TIME={self.time}: agv{agv.id} 准时(time:{self.time})到达运输任务{task_id}的目的地，任务最早完成时间{earliest_time}，任务最晚完成时间{latest_time}')
        elif self.time < earliest_time:

            agv.wait_time = earliest_time
            agv.doing_task_progress = 3
            if self.time_needed:
                penalty = (earliest_time - self.time) // 100.0 * 10.0
                # self.task_to_errly_penalty -= penalty
                self.agv_reward -= penalty
            self.total_task_early += 1
            self.logger.info(f'TIME={self.time}: agv{agv.id} 过早(time:{self.time})到达运输任务{task_id}的目的地，开始等待,任务最早完成时间{earliest_time}')
            return False
        elif self.time > latest_time:
            if self.time_needed:
                penalty = (self.time - latest_time) // 100.0 * 10.0
                # self.task_to_late_penalty -= penalty
                self.agv_reward -= penalty
            self.total_task_late += 1
            self.logger.info(f'TIME={self.time}: agv{agv.id} 过晚(time:{self.time})到达运输任务{task_id}的目的地，任务最晚完成时间{latest_time}')
        
        nums_list = self.get_nums_from_task(task_id)
        place.AGV_in(agv.GLOBAL_ID, nums_list)
        task['state'] = 2  # 任务完成
        task['end time'] = self.time  # 任务完成时间
        # agv.study_mark = True

        mean_time = self.time - task['task create time']
        self.total_calculate_num += 1 
        self.total_task_mean_time = self.total_task_mean_time + 1/self.total_calculate_num * (mean_time - self.total_task_mean_time)

        # self.logger.info(f'TIME={self.time}: 运输任务{task_id} 完成')

    def agv_loading_setting(self, AGV_GLOBAL_ID, place_GLOBAL_ID):
        """place 包含 物料仓、产线、装配台"""
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        place = self.entity_manager.get_entity_by_id(place_GLOBAL_ID)
        task_id = agv.doing_task_id
        nums_list = self.get_nums_from_task(task_id)
        place.AGV_out(agv.GLOBAL_ID, nums_list)

    def task_allocation(self, AGV_GLOBAL_ID, TASK_GLOBAL_ID):
        '''分配AGV运输任务'''
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        task = self.task_manager.get_task_by_id(TASK_GLOBAL_ID)
        start_x, start_y = task['starting point']
        end_x, end_y = task['ending point']
        if agv.accept_task:
            if agv.check_task(start_x, start_y, end_x, end_y):
                self.select_reward += 10  # 任务分配失败惩罚 1
                agv.todo_list.append(TASK_GLOBAL_ID)
                task['AGV_id'] = agv.id
                task['state'] = 1  # 任务状态已分配
                self.unassigned_task.remove(TASK_GLOBAL_ID)
                self.logger.info(f'TIME={self.time}: 任务{TASK_GLOBAL_ID} 由 agv{agv.id}执行')
        else:
            self.select_reward -= 1  # 任务分配失败惩罚 1
            # self.logger.warning(f'TIME={self.time}: agv{agv.id} 无法执行, 运输任务{TASK_GLOBAL_ID} 任务分配失败，需要重新分配')
    
    def get_global_id_by_pos(self, pos):
        for key, val in self.entity_pos_in.items():
            if val == pos:
                return key
        for key, val in self.entity_pos_out.items():
            if val == pos:
                return key
        self.logger.error(f'TIME={self.time}: 未找到{pos}对应的实体')
        return False

    def update_data(self):
        """
        更新数据、包括 生产线/装配台的原材料、成品
        其中原材料用于agent状态，成品用于生成任务
        """
        # 原材料
        self.pLine1_raw_materials = self.produce_line1.raw_material_num
        self.pLine2_raw_materials = self.produce_line2.raw_material_num
        self.pLine3_raw_materials = self.produce_line3.raw_material_num
        
        self.assembly1_raw_materials = self.assembly_stand1.raw_material_num
        self.assembly2_raw_materials = self.assembly_stand2.raw_material_num


        # 成品
        self.pLine1_nums = self.produce_line1.total_production - self.pLine1_nums_used
        self.pLine2_nums = self.produce_line2.total_production - self.pLine2_nums_used
        self.pLine3_nums = self.produce_line3.total_production - self.pLine3_nums_used
        self.ass1_nums = self.assembly_stand1.total_finished_goods - self.ass1_nums_used
        self.ass2_nums = self.assembly_stand2.total_finished_goods - self.ass2_nums_used
        self.ass1_in1_nums = self.assembly_stand1.remain_1
        self.ass1_in2_nums = self.assembly_stand1.remain_2
        self.ass1_in3_nums = self.assembly_stand1.remain_3
        self.ass2_in1_nums = self.assembly_stand2.remain_1
        self.ass2_in2_nums = self.assembly_stand2.remain_2
        self.ass2_in3_nums = self.assembly_stand2.remain_3

    def travel_time(self, starting_point, ending_point):
        sx, sy = starting_point
        ex, ey = ending_point
        dis = L1(sx, sy, ex, ey)
        return dis / self.agv1.speed
    
    def generate_task(self, task_category, products_num=0):

        # self.agv_reward += 1.0
        global TASK_ID

        if task_category == 1:
            # 从物料仓到产线1
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.material_storage.out_pos,
                                        ending_point=self.produce_line1.in_pos, category=task_category, a=4*products_num, b=2*products_num, c=2*products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.material_storage.NAME}到{self.produce_line1.NAME} 运输 a:{4*products_num}, b:{2*products_num}, c:{2*products_num}.')
            self.unassigned_task.append(TASK_ID)
            
        elif task_category == 2:
            # 从物料仓到产线2
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.material_storage.out_pos,
                                        ending_point=self.produce_line2.in_pos, category=task_category, a=2*products_num, b=1*products_num, d=1*products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.material_storage.NAME}到{self.produce_line2.NAME} 运输 a:{2*products_num}, b:{1*products_num}, d:{1*products_num}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 3:
            # 从物料仓到产线3
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.material_storage.out_pos,
                                        ending_point=self.produce_line3.in_pos, category=task_category, a=2*products_num, b=2*products_num, d=1*products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.material_storage.NAME}到{self.produce_line3.NAME} 运输 a:{2*products_num}, b:{2*products_num}, d:{1*products_num}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 4:
            # 从生产线1到装配台1
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line1.out_pos,
                                        ending_point=self.assembly_stand1.in_pos, category=task_category, f1=2*products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line1.NAME}到{self.assembly_stand1.NAME} 运输 f1{2*products_num}.')
            self.unassigned_task.append(TASK_ID)
        
        elif task_category == 5:
            # 从生产线1到装配台2
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line1.out_pos,
                                        ending_point=self.assembly_stand2.in_pos, category=task_category, f1=2*products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line1.NAME}到{self.assembly_stand2.NAME} 运输 f1{2*products_num}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 6:
            # 从生产线2到装配台1
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line2.out_pos,
                                        ending_point=self.assembly_stand1.in_pos, category=task_category, f2=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line2.NAME}到{self.assembly_stand1.NAME} 运输 f2{products_num}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 7:
            # 从生产线2到装配台2
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line2.out_pos,
                                        ending_point=self.assembly_stand2.in_pos, category=task_category, f2=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line2.NAME}到{self.assembly_stand2.NAME} 运输 f2{products_num}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 8:
            # 从生产线3到装配台1
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line3.out_pos,
                                        ending_point=self.assembly_stand1.in_pos, category=task_category, f3=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line3.NAME}到{self.assembly_stand1.NAME} 运输 f3{products_num}.')
            self.unassigned_task.append(TASK_ID)
        
        elif task_category == 9:
            # 从生产线3到装配台2
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.produce_line3.out_pos,
                                        ending_point=self.assembly_stand2.in_pos, category=task_category, f3=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.produce_line3.NAME}到{self.assembly_stand2.NAME} 运输 f3{products_num}.')
            self.unassigned_task.append(TASK_ID)
        
        elif task_category == 10:
            # 从装配台1到成品存储区
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.assembly_stand1.out_pos,
                                        ending_point=self.finishgoods_storage.in_pos, category=task_category, goods=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.assembly_stand1.NAME}到{self.finishgoods_storage.NAME} 运输 goods:{products_num}.')
            self.unassigned_task.append(TASK_ID)
        
        elif task_category == 11:
            # 从装配台2到成品存储区
            TASK_ID += 1
            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, starting_point=self.assembly_stand2.out_pos,
                                        ending_point=self.finishgoods_storage.in_pos, category=task_category, goods=products_num)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.assembly_stand2.NAME}到{self.finishgoods_storage.NAME} 运输 goods:{products_num}.')
            self.unassigned_task.append(TASK_ID)

        
        elif task_category == 12:
            # 从工具仓到装配台1
            TASK_ID += 1
            earliest_arrival_time = self.time

            # 如果最晚到达时间小于路途时间，则更新最晚到达时间
            travel_time = self.time + self.travel_time(self.tools_storage.out_pos, self.assembly_stand1.in_pos)
            travel_time += np.random.randint(100, 200)  # 装卸载、路途时间
            latest_arrival_time = travel_time

            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, earliest_time=earliest_arrival_time,
                                        latest_time=latest_arrival_time, starting_point=self.tools_storage.out_pos,
                                        ending_point=self.assembly_stand1.in_pos, category=task_category, tools=1)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.tools_storage.NAME}到{self.assembly_stand1.NAME} 任务最早到达时间{earliest_arrival_time} 任务最晚到达时间{latest_arrival_time}.')
            self.unassigned_task.append(TASK_ID)

        elif task_category == 13:
            # 从工具仓到装配台2
            TASK_ID += 1
            earliest_arrival_time = self.time

            # 如果最晚到达时间小于路途时间，则更新最晚到达时间
            travel_time = self.time + self.travel_time(self.tools_storage.out_pos, self.assembly_stand2.in_pos)
            travel_time += np.random.randint(100, 200)  # 装卸载、路途时间
            latest_arrival_time = travel_time

            self.task_manager.create_task(task_ID=TASK_ID, task_create_time=self.time, earliest_time=earliest_arrival_time,
                                        latest_time=latest_arrival_time, starting_point=self.tools_storage.out_pos,
                                        ending_point=self.assembly_stand2.in_pos, category=task_category, tools=1)
            self.logger.info(f'TIME={self.time}: 运输任务{TASK_ID}: 从{self.tools_storage.NAME}到{self.assembly_stand2.NAME} 任务最早到达时间{earliest_arrival_time} 任务最晚到达时间{latest_arrival_time}.')
            self.unassigned_task.append(TASK_ID)

        return TASK_ID

        task = self.task_manager.get_task_by_id(TASK_ID)
        return task

    def step(self, time):
        self.time = time
        self.agv1.step(time)
        self.agv2.step(time)
        self.agv3.step(time)
        self.agv4.step(time)
        self.agv5.step(time)
        self.material_storage.step(time)
        self.produce_line1.step(time)
        self.produce_line2.step(time)
        self.produce_line3.step(time)
        self.assembly_stand1.step(time)
        self.assembly_stand2.step(time)
        self.finishgoods_storage.step(time)
        self.update_data()

        num1 = self.pLine1_raw_materials
        num2 = self.pLine2_raw_materials
        num3 = self.pLine3_raw_materials

        global ORDER_ID, MAX_ORDER_NUM
        # 生成订单 
        if self.time > ORDER_GENERATION_INTERVAL + self.order_last_generation and ORDER_ID <= MAX_ORDER_NUM:    
            self.generate_time = self.time + np.random.randint(0, 100)
        if self.time == self.generate_time:
            earlist = self.generate_time + np.random.randint(100, 300)
            lastest = earlist + np.random.randint(500, 1000)
            products = np.random.randint(1, 10)
            id = self.generate_orders(earlist, lastest, products)
            self.order_last_generation = self.generate_time
            
        # 检查订单，更新订单状态
        self.check_order()
        
        # 处理未完成的订单，根据订单状态生成运输任务
        self.process_order()

    def check_order(self):
        for i in self.order_doing_list:
            order = self.order_manager.get_order_by_id(i)
            task_id_list = order['task_id']
            for task_id in task_id_list:
                task = self.task_manager.get_task_by_id(task_id)
                if task['state'] == 2:
                    order['task_id'].remove(task_id)
            if len(order['task_id']) == 0:
                order['doing'] = False

            
    def process_order(self):
        global MAX_ORDER_NUM
        # 处理未完成的订单，根据订单状态生成运输任务
        for i in self.order_doing_list:
            order = self.order_manager.get_order_by_id(i)
            # 生成第一阶段运输任务
            if order['state'] == 0 and order['doing'] == False:
                task1_id = self.generate_task(1, order['products'])
                order['task_id'].append(task1_id)
                order['task_id'].append(self.generate_task(2, order['products']))
                order['task_id'].append(self.generate_task(3, order['products']))
                order['task1_id'] = task1_id
                order['task4_id'] = task1_id
                order['state'] = 1
                order['doing'] = True

            # 生成第二阶段运输任务
            if order['state'] == 1 and order['doing'] == False:
                if order['category'] == 0:
                    task2_id = self.generate_task(4, order['products'])
                    order['task_id'].append(task2_id)
                    order['task_id'].append(self.generate_task(6, order['products']))
                    order['task_id'].append(self.generate_task(8, order['products']))
                else:
                    task2_id = self.generate_task(5, order['products'])
                    order['task_id'].append(task2_id)
                    order['task_id'].append(self.generate_task(7, order['products']))
                    order['task_id'].append(self.generate_task(9, order['products']))
                order['task2_id'] = task2_id
                order['task5_id'] = task2_id
                order['state'] = 2
                order['doing'] = True

            # 生成第三段运输任务
            if order['state'] == 2 and order['doing'] == False:
                if order['category'] == 0:
                    task3_id = self.generate_task(10, order['products'])
                    order['task_id'].append(task3_id)
                else:
                    task3_id = self.generate_task(11, order['products'])
                    order['task_id'].append(task3_id)
                order['task3_id'] = task3_id
                order['state'] = 3
                order['doing'] = True

            if order['state'] == 3 and order['doing'] == False:
                order['state'] = 4
                order['finishtime'] = self.time 
                self.logger.info(f'TIME={self.time}: 订单完成{order["order_ID"]}.') 

                if self.time < order['lastest'] and self.time > order['earlist']:
                    self.order_intime += 1
                    self.agv_reward += 1000
                    self.select_reward += 1000

                else:
                    self.agv_reward -= 1000
                    self.select_reward -= 1000 


                global MAX_ORDER_NUM
                if order['order_ID'] == MAX_ORDER_NUM:
                    self.logger.warning(f'TIME={self.time}: 所有订单均完成.') 
                    self.done = True
                order['doing'] == True
            
        return self.order_intime

    @property
    def get_task_total_nums(self):
        global TASK_ID
        return TASK_ID

    def reset_task(self):
        global TASK_ID
        TASK_ID = 0
    
    def reset_order(self):
        global ORDER_ID
        ORDER_ID = 0

    def change_order_num(self, order_num):
        global MAX_ORDER_NUM
        MAX_ORDER_NUM = order_num


class AGV:                            
    """AGV"""
    def __init__(self, range_limit, volume_limit, weight_limit, speed, id, GLOBAL_ID, task_manager, control_center, logger=None):
        self.position_x = AGV_ORIGINAL_X  # AGV的X坐标
        self.position_y = AGV_ORIGINAL_Y  # AGV的Y坐标
        self.range_limit = range_limit  # 里程上限
        self.remain_range = range_limit  # 剩余续航里程
        self.volume_limit = volume_limit  # 体积上限
        self.weight_limit = weight_limit  # 重量上限
        self.speed = speed  # 速度
        self.volume_loaded = 0  # 已装载货物体积
        self.weight_loaded = 0  # 已装载货物重量
        self.back_range = 0  # 距离AGV充电区的距离
        self.remain_A = 0
        self.remain_B = 0
        self.remain_C = 0
        self.remain_D = 0
        self.remain_1 = 0
        self.remain_2 = 0
        self.remain_3 = 0
        self.remain_goods = 0
        self.remain_tools = 0
        self.id = id
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = f'AGV_{self.id}'
        self.remain_travel_time = 0  # 剩余行驶时间
        self.remain_load_time = 0  # 剩余装载时间
        self.remain_unload_time = 0  # 剩余卸载时间
        self.remain_charging_time = 0  # 剩余充电时间
        self.state = 0  # 0:等待中，1：装载 2：卸载 3：运输途中 4：充电
        self.logger = logger
        self.doing_task_id = 0  # 正在处理任务列表
        self.todo_list = []  # 待办任务列表
        self.task_manager = task_manager
        self.control_center = control_center
        self.doing_task_progress = 0  # 任务进展指示，0 未开始， 1 已到达起点， 2 已完成装载， 3 已到达终点，4 已卸载，任务完成
        self.time = 0
        self.accept_task = True  # 是否接受任务， 当电量低时 不接受
        self.remain_mileage = 0  # 已经行驶里程, 包含当前位置到任务起点，任务起点到任务终点
        self.wait_time = 0
        self.study_mark = True
        self.working_time = 0  # 工作时间
        self.task_nums = []

    def check_task(self, start_x, start_y, end_x, end_y):
        # if len(self.todo_list) == 5:
        #     # self.logger.warning(f'TIME={self.time}: AGV{self.id} 待处理任务数已达5个，不能再接新任务')
        #     return False
        # return True

        total_dis = 0 
        total_dis += 2 * L1(start_x, start_y, end_x, end_y)
        # print(self.todo_list)
        for task_id in self.todo_list:
            task = self.task_manager.get_task_by_id(task_id)
            start_x, start_y = task['starting point']
            end_x, end_y = task['ending point']
            total_dis += 2 * L1(start_x, start_y, end_x, end_y)
        total_dis += L1(self.position_x, self.position_y, AGV_ORIGINAL_X, AGV_ORIGINAL_Y)
        total_dis += 2000.

        if total_dis <= self.remain_range:
            if len(self.todo_list) < 5:
                return True
            else:
                return False
        else:
            self.accept_task = False
            if -1 not in self.todo_list:
                self.logger.info(f'TIME={self.time}: AGV{self.id} 剩余续航里程{self.remain_range}，停止接受任务，目前剩余{len(self.todo_list)}件任务') 
                self.todo_list.append(-1)  # 添加任务id为-1的任务，为充电任务
            return False

    def check_cargo(self):
        volum_sum = self.remain_A * METRIAL_A_VOLUME + self.remain_B * METRIAL_B_VOLUME + \
                    self.remain_C * METRIAL_C_VOLUME + self.remain_D * METRIAL_D_VOLUME + \
                    self.remain_1 * SEMI_FINISHED_1_VOLUME + self.remain_2 * SEMI_FINISHED_2_VOLUME + \
                    self.remain_3 * SEMI_FINISHED_3_VOLUME + self.remain_goods * FINISHED_GOODS_VOLUME
        # weight_sum = self.remain_A * METRIAL_A_WEIGHT + self.remain_B * METRIAL_B_WEIGHT + \
        #             self.remain_C * METRIAL_C_WEIGHT + self.remain_D * METRIAL_D_WEIGHT +self.remain_1 * SEMI_FINISHED_1_WEIGHT + \
        #             self.remain_2 * SEMI_FINISHED_2_WEIGHT + self.remain_3 * SEMI_FINISHED_3_WEIGHT + \
        #             self.remain_goods * FINISHED_GOODS_WEIGHT
        if volum_sum > self.volume_limit:
            self.logger.critical(f'TIME={self.time}: AGV{self.id} 装载失败，超出体积限制')
        # if weight_sum > self.weight_limit:
        #     logger.critical(f'AGV{self.id} 装载失败，超出重量限制')
        # if volum_sum <= self.volume_limit and weight_sum <= self.weight_limit:
        else:
            self.logger.info(f'TIME={self.time}: AGV{self.id} 装载成功，当前装载 A:{self.remain_A} B:{self.remain_B} C:{self.remain_C} D:{self.remain_D} '
                        f'半成品1:{self.remain_1} 半成品2:{self.remain_2} 半成品3:{self.remain_3} 成品:{self.remain_goods}')
    
    def set_destination(self, end_x, end_y):
        distance = L1(self.position_x, self.position_y, end_x, end_y)
        if distance > self.remain_range:
            self.logger.warning(f'TIME={self.time}: AGV{self.id} 设置目的地失败，距离目的地路程超出剩余续航里程')
            # self.control_center.reward -= 1.0
        else:
            travel_time = int(distance / self.speed)
            self.state = 3
            self.remain_travel_time = travel_time
            self.remain_range -= distance  # 剩余里程 减去 该段路程
            self.position_x = end_x
            self.position_y = end_y
            self.logger.info(f'TIME={self.time}: AGV{self.id} 设置目的地成功，预计行驶{travel_time}t')

    def begin_charging(self):
        self.remain_charging_time = (self.range_limit - self.remain_range) / self.range_limit * AGV_CHARGING_TIME 
        self.state = 4  # 设置状态为充电中
        self.logger.info(f'TIME={self.time}: AGV{self.id} 开始充电，预计充电时间{self.remain_charging_time}t, 剩余未完成任务数{len(self.todo_list)}')

    def begin_load(self):
        self.remain_load_time = AGV_LOAD_TIME
        self.state = 1  # 设置状态为装载中
        self.logger.info(f'TIME={self.time}: AGV{self.id} 开始装载货物，装载时间{self.remain_load_time}t')
    
    def begin_unload(self):
        self.remain_unload_time = AGV_UNLOAD_TIME
        self.state = 2  # 设置状态为装载中
        self.logger.info(f'TIME={self.time}: AGV{self.id} 开始卸载货物，卸载时间{self.remain_unload_time}t')  

    def step(self, time):
        self.time = time
        self.task_nums.append(len(self.todo_list))
        if self.state != 0:
            self.working_time += 1

        if self.time < self.wait_time:
            return

        # if self.remain_range < (0.2 * self.range_limit) and self.accept_task == True:  # 电量少于0.25
        #     self.accept_task = False
        #     self.logger.info(f'TIME={self.time}: AGV{self.id} 剩余电量小于20%，停止接受任务，目前剩余{len(self.todo_list)}件任务') 
        #     self.todo_list.append(-1)  # 添加任务id为-1的任务，为充电任务

        # if self.state == 0 and self.doing_task_id == 0 and len(self.todo_list) != 0:
        #     self.doing_task_id = self.todo_list[0]
        #     self.todo_list.remove(self.doing_task_id)
        #     self.logger.info(f'TIME={self.time}: AGV{self.id} 开始执行任务{self.doing_task_id}')

        if self.state == 0 and self.doing_task_id!=0:  # 如果状态为 0 且 有任务待处理
            
            if self.doing_task_id == -1:  # 充电任务
                if self.doing_task_progress == 0:
                    self.set_destination(AGV_ORIGINAL_X, AGV_ORIGINAL_Y)
                    self.doing_task_progress = 1
                elif self.doing_task_progress == 1:
                    self.begin_charging()
                    self.doing_task_progress = 2
                elif self.doing_task_progress == 2:
                    self.accept_task = True  # 充电完成，可以接受任务
                    self.study_mark = True
                    self.doing_task_id = 0
                    self.doing_task_progress = 0
                    self.logger.info(f'TIME={self.time}: AGV{self.id} 开始接受任务') 

            else:
                task = self.task_manager.get_task_by_id(self.doing_task_id)
                if self.doing_task_progress == 0:  # 任务未开始
                    start_x, start_y = task['starting point']
                    self.set_destination(start_x, start_y)
                    self.doing_task_progress = 1
                elif self.doing_task_progress == 1:  # 已到达起点
                    pos = task['starting point']
                    place_id = self.control_center.get_global_id_by_pos(pos)
                    self.control_center.agv_loading_setting(self.GLOBAL_ID, place_id)
                    self.doing_task_progress = 2
                elif self.doing_task_progress == 2:  # 已完成装载
                    end_x, end_y = task['ending point']
                    self.set_destination(end_x, end_y)
                    self.doing_task_progress = 3
                elif self.doing_task_progress == 3:  # 已到达终点
                    pos = task['ending point']
                    place_id = self.control_center.get_global_id_by_pos(pos)
                    self.doing_task_progress = 4
                    self.control_center.agv_unloading_setting(self.GLOBAL_ID, place_id)
                elif self.doing_task_progress == 4:  # 已完成卸载，任务完成
                    task['state'] = 2
                    # self.study_mark = True
                    self.doing_task_id = 0
                    self.doing_task_progress = 0
                    self.study_mark = True
                    self.logger.info(f"TIME={self.time}: 运输任务{task['task_ID']}完成")
                    

        if self.state == 1:  # 状态为装载中
            self.remain_load_time -= 1
            if self.remain_load_time <= 0:
                self.state = 0
                self.logger.info(f'TIME={self.time}: AGV{self.id} 装载货物完成.')
                self.logger.info(f'TIME={self.time}: AGV{self.id} 当前装载 A:{self.remain_A} B:{self.remain_B} C:{self.remain_C} D:{self.remain_D} '
                        f'半成品1:{self.remain_1} 半成品2:{self.remain_2} 半成品3:{self.remain_3} 成品:{self.remain_goods}')
        
        if self.state == 2:  # 状态为卸载中
            self.remain_unload_time -= 1
            if self.remain_unload_time <= 0:
                self.state = 0
                self.logger.info(f'TIME={self.time}: AGV{self.id} 卸载货物完成.')
                self.logger.info(f'TIME={self.time}: AGV{self.id} 当前装载 A:{self.remain_A} B:{self.remain_B} C:{self.remain_C} D:{self.remain_D} '
                        f'半成品1:{self.remain_1} 半成品2:{self.remain_2} 半成品3:{self.remain_3} 成品:{self.remain_goods}')

        if self.state == 3:  # 状态为行驶中
            self.remain_travel_time -= 1
            if self.remain_travel_time <= 0:
                self.state = 0  # 设置AGV状态为等待操作
                self.logger.info(f'TIME={self.time}: AGV{self.id} 到达行驶目的地.')

        if self.state == 4:  # 状态为充电中
            self.remain_charging_time -= 1
            if self.remain_charging_time <= 0:
                self.state = 0  # 设置AGV状态为等待操作
                self.remain_range = self.range_limit  # 更新剩余里程
                self.logger.info(f'TIME={self.time}: AGV{self.id} 充电完成.')


class ProductionLine:
    """生产线"""
    def __init__(self, in_x, in_y, out_x, out_y, target_num, id, GLOBAL_ID, entity_manager, logger=None, in_volume_limit=800, control_center=None):
        self.in_x = in_x
        self.in_y = in_y
        self.out_x = out_x
        self.out_y = out_y
        self.in_weight_limit = 10000  # 暂不考虑
        self.in_volume_limit = in_volume_limit
        self.out_weight_limit = 10000  # 暂不考虑
        self.out_volume_limit = 800
        self.remain_A = 0
        self.remain_B = 0
        self.remain_C = 0
        self.remain_D = 0
        self.remain_after_production = 0  # 经过加工的产品数量
        self.total_production = 0  # 生产总量
        self.used_production = 0  # 已经被使用的数量
        # self.in_weight_remain = 0
        # self.in_volume_remain = 0
        # self.out_volume_remain = 0
        # self.out_weight_remain = 0
        self.target_num = target_num  # 生产目标，分为 1--半成品1 2--半成品2 3--半成品3
        self.production_list = []
        self.id = id
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = f'ProductionLine_{self.id}'
        self.entity_manager = entity_manager
        self.loading_time = 0  # 装载剩余时间
        self.unloading_time = 0
        self.remain_A_loading = 0  # 正在装载的货物
        self.remain_B_loading = 0
        self.remain_C_loading = 0
        self.remain_D_loading = 0
        self.logger = logger
        self.wait_to_add_time = PRODUCTION_REQUIREMENT[self.target_num][4]
        self.time = 0
        self.working_time = 0
        self.waiting_time = 0
        self.raw_material_num = 0
        self.control_center = control_center

        if self.target_num == 1:
            self.SEMI_FINISHED_VOLUME = SEMI_FINISHED_1_VOLUME
            self.SEMI_FINISHED_WEIGHT = SEMI_FINISHED_1_WEIGHT
        elif self.target_num == 2:
            self.SEMI_FINISHED_VOLUME = SEMI_FINISHED_2_VOLUME
            self.SEMI_FINISHED_WEIGHT = SEMI_FINISHED_2_WEIGHT
        elif self.target_num == 3:
            self.SEMI_FINISHED_VOLUME = SEMI_FINISHED_3_VOLUME
            self.SEMI_FINISHED_WEIGHT = SEMI_FINISHED_3_WEIGHT

    def add_to_production_list(self, nums):
        coby_nums = nums
        while nums > 0:
            if self.remain_A >= PRODUCTION_REQUIREMENT[self.target_num][0] and self.remain_B >= PRODUCTION_REQUIREMENT[self.target_num][1] and \
                self.remain_C >= PRODUCTION_REQUIREMENT[self.target_num][2] and self.remain_D >= PRODUCTION_REQUIREMENT[self.target_num][3]:
                self.production_list.append(PRODUCTION_REQUIREMENT[self.target_num][4])
                self.remain_A -= PRODUCTION_REQUIREMENT[self.target_num][0]
                self.remain_B -= PRODUCTION_REQUIREMENT[self.target_num][1]
                self.remain_C -= PRODUCTION_REQUIREMENT[self.target_num][2]
                self.remain_D -= PRODUCTION_REQUIREMENT[self.target_num][3]
                nums -= 1
                
            else:
                if self.remain_A < PRODUCTION_REQUIREMENT[self.target_num][0]:
                    self.logger.warning(f'TIME={self.time}: 产线{self.id} 缺少物料A.')
                if self.remain_B < PRODUCTION_REQUIREMENT[self.target_num][1]:
                    self.logger.warning(f'TIME={self.time}: 产线{self.id} 缺少物料B.')
                if self.remain_C < PRODUCTION_REQUIREMENT[self.target_num][2]:
                    self.logger.warning(f'TIME={self.time}: 产线{self.id} 缺少物料C.')
                if self.remain_D < PRODUCTION_REQUIREMENT[self.target_num][3]:
                    self.logger.warning(f'TIME={self.time}: 产线{self.id} 缺少物料D.')
                break
        if nums == 0:
            pass
            # self.logger.info(f'产线{self.id} 生产任务添加成功，添加生产半成品{self.target_num} {coby_nums - nums}件.')
        else:
            self.logger.warning(f'TIME={self.time}: 产线{self.id} 生产任务添加失败，计划添加{coby_nums}件，添加成功{coby_nums - nums}件，添加生产半成品{self.target_num}:{coby_nums - nums}件.')

    def update_in_and_out(self):
        in_volume = self.remain_A * METRIAL_A_VOLUME + self.remain_B * METRIAL_B_VOLUME + \
                    self.remain_C * METRIAL_C_VOLUME
        # in_weight = self.remain_A * METRIAL_A_WEIGHT + self.remain_B * METRIAL_B_WEIGHT + \
        #             self.remain_C * METRIAL_C_WEIGHT
        out_volume = self.remain_after_production * self.SEMI_FINISHED_VOLUME
        # out_weight = self.remain_after_production * self.SEMI_FINISHED_WEIGHT
        # if in_weight > self.in_weight_limit or in_volume > self.in_volume_limit or \
        #         out_weight > self.out_weight_limit or out_volume > self.out_volume_limit:
        if in_volume > self.in_volume_limit or out_volume > self.out_volume_limit:
            # if in_weight > self.in_weight_limit:
            #     logger.critical(f'TIME={self.time}: 产线{self.id} 进料仓堆积，超出重量限制')
            if in_volume > self.in_volume_limit:
                self.logger.critical(f'TIME={self.time}: 产线{self.id} 进料仓堆积，超出体积限制')
                #self.control_center.agv_reward -= 50
                # self.control_center.agv_reward += 50 * self.unloading_goods  # 设置成品数目奖励，鼓励AGV完成更多的成品的生产 
            # if out_weight > self.out_weight_limit:
            #     logger.critical(f'TIME={self.time}: 产线{self.id} 出料仓堆积，超出重量限制')
            if out_volume > self.out_volume_limit:
                self.logger.critical(f'TIME={self.time}: 产线{self.id} 出料仓堆积，超出体积限制')   
                #self.control_center.agv_reward -= 50
            return False
        else:
            # self.in_weight_remain = in_weight
            # self.in_volume_remain = in_volume
            # self.out_volume_remain = out_volume
            # self.out_weight_remain = out_weight
            return True
        
    def step(self, time):
        self.time = time

        # 更新原材料剩余可加工半成品件数
        self.raw_material_num = self.remain_A / PRODUCTION_REQUIREMENT[self.target_num][0]

        # 每个步骤均检测一次，如果进料仓有充足物料，则添加一个生产任务，如果出料仓堆积，则不加入生产任务
        if self.remain_A >= PRODUCTION_REQUIREMENT[self.target_num][0] and self.remain_B >= PRODUCTION_REQUIREMENT[self.target_num][1] and \
                self.remain_C >= PRODUCTION_REQUIREMENT[self.target_num][2] and self.remain_D >= PRODUCTION_REQUIREMENT[self.target_num][3]:
            if self.wait_to_add_time <= 0:    
                if self.remain_after_production * self.SEMI_FINISHED_VOLUME <= self.out_volume_limit:
                    self.add_to_production_list(1)
                    self.wait_to_add_time = PRODUCTION_REQUIREMENT[self.target_num][4]
        else:
            if self.time > 500:
                # self.logger.warning(f'TIME={self.time}: 产线{self.id} 缺少原材料，无法继续添加生产任务。')
                pass
        self.wait_to_add_time -= 1
        
        if len(self.production_list) == 0:
            self.waiting_time += 1
        else:
            self.working_time += 1

        zero_index = []
        for i in range(len(self.production_list)):  # 对列表中的每个元素-1t
            self.production_list[i] = self.production_list[i] - 1
        for i in range(len(self.production_list)):  # 再遍历列表处理列表中的少于0的元素值
            if self.production_list[i] <= 0:
                self.remain_after_production += 1
                self.total_production += 1
                zero_index.append(i)
        if zero_index:
            zero_index.reverse()  # 倒叙索引列表，防止修改时元素索引改变
            self.logger.info(f'TIME={self.time}: 产线{self.id} 完成生产{len(zero_index)}件半成品{self.target_num}，现有半成品{self.target_num} {self.remain_after_production}件.')
            for i in range(len(zero_index)):
                self.production_list.pop(zero_index[i])  # 删除小于0的任务
        if self.unloading_time > 0:
            self.unloading_time -= 1
            if self.unloading_time == 0:
                self.remain_A += self.remain_A_loading
                self.remain_B += self.remain_B_loading
                self.remain_C += self.remain_C_loading
                self.remain_D += self.remain_D_loading
                self.remain_A_loading = 0
                self.remain_B_loading = 0
                self.remain_C_loading = 0
                self.remain_D_loading = 0
                if self.update_in_and_out():
                    self.logger.info(f'TIME={self.time}: 产线{self.id} 进料仓物料补充完成，现有 A:{self.remain_A} B:{self.remain_B}' 
                                    f' C:{self.remain_C} D:{self.remain_D}')
                #else:  # 物料超出重量/体积限制
                #    self.control_center.agv_reward -= 10.0
        if self.loading_time > 0:
            self.loading_time -= 1
            if self.loading_time == 0:
                if self.update_in_and_out():
                    self.logger.info(f'TIME={self.time}: 产线{self.id} 出料仓装载AGV任务完成，现有半成品{self.remain_after_production}件.' 
                                    f' C:{self.remain_C} D:{self.remain_D}')
                #else:  # 物料超出重量/体积限制
                #    self.control_center.agv_reward -= 10.0
    def check_agv_in_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.in_x and agv.position_y == self.in_y and agv.state == 0:
            return True
        return False

    @property
    def in_pos(self):
        return self.in_x, self.in_y
    
    @property
    def out_pos(self):
        return self.out_x, self.out_y

    def AGV_in(self, AGV_GLOBAL_ID, nums_list):
        num_a = nums_list.get('a')
        num_b = nums_list.get('b')
        num_c = nums_list.get('c')
        num_d = nums_list.get('d')
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        flag = 0
        if not self.check_agv_in_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达产线{self.id}进料区或AGV状态不在等待中, 卸载失败.')
            return False
        # A
        if agv.remain_A >= num_a:
            self.remain_A_loading += num_a
            agv.remain_A -= num_a
            flag = 1
        # B
        if agv.remain_B >= num_b:
            self.remain_B_loading += num_b
            agv.remain_B -= num_b
            flag = 1
        # C
        if agv.remain_C >= num_c:
            self.remain_C_loading += num_c
            agv.remain_C -= num_c
            flag = 1
        # D
        if agv.remain_D >= num_d:
            self.remain_D_loading += num_d
            agv.remain_D -= num_d
            flag = 1
        if flag:
            self.unloading_time = AGV_UNLOAD_TIME
            agv.begin_unload()
        else:
            self.logger.warning(f'TIME={self.time}: AGV{agv.id} 在产线{self.id}处未实际携带所需物料，卸载失败')
        

    def AGV_out(self, AGV_GLOBAL_ID, nums_list):
        nums_of_goods = nums_list.get(str(self.target_num))
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if nums_of_goods > self.remain_after_production:
            self.logger.error(f'TIME={self.time}: 产线{self.id} 现有半成品{self.target_num}: {self.remain_after_production}件少于指定卸载数量{nums_of_goods}件, AGV装载任务失败.')
            return False
        if self.target_num == 1:
            agv.remain_1 += nums_of_goods
        elif self.target_num == 2:
            agv.remain_2 += nums_of_goods
        elif self.target_num == 3:
            agv.remain_3 += nums_of_goods 
        self.remain_after_production -= nums_of_goods
        agv.begin_load()
        self.loading_time = AGV_LOAD_TIME


class MaterialStorage:
    """物料仓"""
    def __init__(self, GLOBAL_ID, entity_manager, original_A=800 , original_B=500, original_C=300, 
                 original_D=300, update_time=3000, pos_x=300, pos_y=100, logger=None):
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = 'MaterialStorage'
        self.entity_manager = entity_manager
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.original_A = original_A
        self.original_B = original_B
        self.original_C = original_C
        self.original_D = original_D
        self.remain_A = original_A
        self.remain_B = original_B
        self.remain_C = original_C
        self.remain_D = original_D
        self.update_time = update_time  # 每隔3000t物料仓补货一次
        self.remain_time = update_time
        self.logger = logger
        self.time = 0

    def step(self, time):
        self.time = time
        self.remain_time -= 1
        if self.remain_time <= 0:
            self.remain_time = self.update_time
            self.remain_A += self.original_A
            self.remain_B += self.original_B
            self.remain_C += self.original_C
            self.remain_D += self.original_D
            self.logger.info(f'TIME={self.time}: 物料仓 补货完成，更新后 A:{self.remain_A} B:{self.remain_B} C:{self.remain_C} D:{self.remain_D}')
    
    def check_agv_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.pos_x and agv.position_y == self.pos_y and agv.state == 0:
            return True
        return False

    def AGV_out(self, AGV_GLOBAL_ID, nums_list):
        num_a = nums_list.get('a')
        num_b = nums_list.get('b')
        num_c = nums_list.get('c')
        num_d = nums_list.get('d')
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if not self.check_agv_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达物料仓或AGV状态不在等待中,装载失败.')
            return False
        if num_a > self.remain_A:
            self.logger.error(f'TIME={self.time}: 物料仓 现有 原材料A: {self.remain_A}件少于指定装载{num_a}件,装载失败')
            num_a = 0 
        else:
            agv.remain_A += num_a
            self.remain_A -= num_a
        if num_b > self.remain_B:
            self.logger.error(f'TIME={self.time}: 物料仓 现有 原材料B: {self.remain_B}件少于指定装载{num_b}件,装载失败')
            num_b = 0
        else:
            agv.remain_B += num_b
            self.remain_B -= num_b
        if num_c > self.remain_C:
            self.logger.error(f'TIME={self.time}: 物料仓 现有 原材料C: {self.remain_C}件少于指定装载{num_c}件,装载失败')
            num_c = 0
        else:
            agv.remain_C += num_c
            self.remain_C -= num_c
        if num_d > self.remain_D:
            self.logger.error(f'TIME={self.time}: 物料仓 现有 原材料D: {self.remain_D}件少于指定装载{num_d}件,装载失败')
            num_d = 0
        else:
            agv.remain_D += num_d
            self.remain_D -= num_d
        agv.begin_load()
        self.logger.info(f'TIME={self.time}: AGV{agv.id} 从物料仓 装载物料 A:{num_a} B:{num_b} C:{num_c} D:{num_d}')
    
    @property
    def out_pos(self):
        return self.pos_x, self.pos_y
        

class ToolStorage:
    """工具仓"""
    def __init__(self, GLOBAL_ID, entity_manager, pos_x=600, pos_y=100, logger=None, control_center=None):
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = 'ToolStorage'
        self.entity_manager = entity_manager
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.remain_tools = 10000  # 工具仓的工具数目很多...
        self.loading_time = 0
        self.logger = logger
        self.time = 0
        self.control_center = control_center

    def step(self, time):
        self.time = time
        if self.loading_time > 0:
            self.loading_time -= 1
            if self.loading_time == 0:
                pass
    
    def check_agv_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.pos_x and agv.position_y == self.pos_y and agv.state == 0:
            return True
        return False

    def AGV_out(self, AGV_GLOBAL_ID, num_list=[]):
        num_tools = num_list.get('tools')
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if not self.check_agv_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达工具仓或AGV状态不在等待中,装载失败.')
            return False
        agv.remain_tools = num_tools
        self.logger.info(f'TIME={self.time}: AGV{agv.id}在工具仓装载工具{num_tools}个')
        agv.begin_load()
        self.loading_time = AGV_LOAD_TIME

    @property
    def out_pos(self):
        return self.pos_x, self.pos_y

class AssemblyStand:
    """装配台"""
    def __init__(self, in_x, in_y, out_x, out_y, id, GLOBAL_ID, entity_manager, logger=None, control_center=None):
        self.in_x = in_x
        self.in_y = in_y
        self.out_x = out_x
        self.out_y = out_y
        self.id = id
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = f'AssemblyStand_{self.id}'
        self.entity_manager = entity_manager
        self.remain_1 = 0
        self.remain_2 = 0
        self.remain_3 = 0
        self.remain_finished_goods = 0  # 成品数
        self.total_finished_goods = 0  # 成品数
        self.assembly_list = []  # 待装配列表
        self.remain_1_loading = 0  # 正在装载的半成品
        self.remain_2_loading = 0 
        self.remain_3_loading = 0 
        self.unloading_time = 0 
        self.loading_time = 0 
        self.logger = logger 
        self.time = 0 
        self.wait_to_add_time = 0
        self.working_time = 0
        self.waiting_time = 0
        self.raw_material_num = [0, 0, 0]
        self.lack_raw_1 = False
        self.lack_raw_2 = False
        self.lack_raw_3 = False
        self.lack_raw_1_times = 0
        self.lack_raw_2_times = 0
        self.lack_raw_3_times = 0
        self.ask_to_transport = False
        self.ask_to_transport_times = 0
        self.tools_num = 5
        self.assem_speed = 1.0
        self.lmbda = -np.log(1 - 0.01) / 10000  # 在50000的时候的损坏概率为0.05
        self.failure_probablities = 0
        self.control_center = control_center

    
    def exponential_pdf(self, x, lmbda):
        """
        计算指数分布的概率密度函数值

        Parameters:
            x (numpy.ndarray): 时间点
            lmbda (float): 分布的速率参数

        Returns:
            numpy.ndarray: 指数分布的概率密度函数值
        """
        return lmbda * np.exp(-lmbda * x)
               
    
    def check_agv_in_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.in_x and agv.position_y == self.in_y and agv.state == 0:
            return True
        return False

    def check_agv_out_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.out_x and agv.position_y == self.out_y and agv.state == 0:
            return True
        return False

    def add_to_assembly_list(self, nums):
        copy_nums = nums
        while nums > 0:
            if self.remain_1 >= ASSEMBLY_REQUIREMENT[0] and self.remain_2 >= ASSEMBLY_REQUIREMENT[1]  \
                and self.remain_3 >= ASSEMBLY_REQUIREMENT[2]:
                self.assembly_list.append(ASSEMBLY_REQUIREMENT[3])
                self.remain_1 -= ASSEMBLY_REQUIREMENT[0]
                self.remain_2 -= ASSEMBLY_REQUIREMENT[1]
                self.remain_3 -= ASSEMBLY_REQUIREMENT[2]
                nums -= 1
            else:
                if self.remain_1 < ASSEMBLY_REQUIREMENT[0]:
                    self.logger.warning(f'TIME={self.time}: 装配台{self.id} 缺少半成品1')
                if self.remain_2 < ASSEMBLY_REQUIREMENT[1]:
                    self.logger.warning(f'TIME={self.time}: 装配台{self.id} 缺少半成品2')
                if self.remain_3 < ASSEMBLY_REQUIREMENT[2]:
                    self.logger.warning(f'TIME={self.time}: 装配台{self.id} 缺少半成品3')
                break
        if nums == 0:
            self.logger.info(f'TIME={self.time}: 装配台{self.id} 装配任务添加成功，添加装配成品{copy_nums - nums}件')
        else:
            self.logger.warning(f'TIME={self.time}: 装配台{self.id} 装配任务添加失败，计划添加{copy_nums}件，添加成功{copy_nums - nums}件')

    def step(self, time):
        self.time = time
        
        self.failure_probablities = 1 - np.exp(-self.lmbda * self.time)

        # if self.tools_num == 0:
            # self.control_center.reward -= 1.0

        if self.time > 5000 and self.time % 100 == 0:
            # if np.random.rand() < self.failure_probablities and self.tools_num > 0:
            if np.random.rand() < 0.00 and self.tools_num > 0:
                self.tools_num -= 1
                self.assem_speed = 1. * self.tools_num / 5
                if self.id == 1:
                    self.control_center.generate_task(12)
                if self.id == 2:
                    self.control_center.generate_task(13)
                self.logger.warning(f'TIME={self.time}: 装配台{self.id} 出现工具损坏，剩余可用工具数目为{self.tools_num}个，装配速度为{self.assem_speed}倍')

        # 更新原材料数目
        self.raw_material_num[0] = self.remain_1 / ASSEMBLY_REQUIREMENT[0]
        self.raw_material_num[1] = self.remain_2 / ASSEMBLY_REQUIREMENT[1]
        self.raw_material_num[2] = self.remain_3 / ASSEMBLY_REQUIREMENT[2]

        # if self.time > 2000:
        if self.raw_material_num[0] <= 10:
            if self.lack_raw_1_times == 0:
                self.lack_raw_1 = True
                self.lack_raw_1_times = ASS_WAIT_TIME
            else:
                self.lack_raw_1_times -= 1
        else:
            self.lack_raw_1 = False
            self.lack_raw_1_times = 0
        
        if self.raw_material_num[1] <= 10:
            if self.lack_raw_2_times == 0:
                self.lack_raw_2 = True
                self.lack_raw_2_times =  ASS_WAIT_TIME
            else:
                self.lack_raw_2_times -=  1
        else:
            self.lack_raw_2 = False
            self.lack_raw_2_times = 0

        if self.raw_material_num[2] <= 10:
            if self.lack_raw_3_times ==  0:
                self.lack_raw_3 = True
                self.lack_raw_3_times =  ASS_WAIT_TIME
            else:
                self.lack_raw_3_times -=  1
        else:
            self.lack_raw_3 = False
            self.lack_raw_3_times = 0

        if self.remain_finished_goods > 5:
            if self.ask_to_transport_times == 0:
                self.ask_to_transport = True
                self.ask_to_transport_times = ASS_WAIT_TIME
            else:
                self.ask_to_transport_times -=  1
        else:
            self.ask_to_transport = False
            self.ask_to_transport_times = 0
        

        # 每步均检查一次，如果进料仓原材料充足，则添加一个装配任务
        if self.remain_1 >= ASSEMBLY_REQUIREMENT[0] and self.remain_2 >= ASSEMBLY_REQUIREMENT[1]  \
            and self.remain_3 >= ASSEMBLY_REQUIREMENT[2]:
            if self.wait_to_add_time <= 0:
                self.add_to_assembly_list(1)
                self.wait_to_add_time = ASSEMBLY_REQUIREMENT[3]
        self.wait_to_add_time -= 1

        if len(self.assembly_list) == 0 or self.tools_num == 0:
            self.waiting_time += 1
        else:
            self.working_time += 1
        
        zero_index = []
        for i in range(len(self.assembly_list)):  # 对列表中的每个元素 -1t
            self.assembly_list[i] -= 1 * self.assem_speed
        for i in range(len(self.assembly_list)):  # 再遍历列表处理列表中的少于0的元素值
            if self.assembly_list[i] <= 0:
                self.remain_finished_goods += 1
                self.total_finished_goods += 1
                zero_index.append(i)
        if zero_index:
            zero_index.reverse()  # 倒叙索引列表，防止修改时元素索引改变
            self.logger.info(f'TIME={self.time}: 装配台{self.id} 完成装配{len(zero_index)}件成品, 目前成品存量{self.remain_finished_goods}')
            for i in range(len(zero_index)):
                self.assembly_list.pop(zero_index[i])  # 删除小于0的任务
        if self.unloading_time > 0:
            self.unloading_time -= 1
            if self.unloading_time == 0:
                self.remain_1 += self.remain_1_loading
                self.remain_2 += self.remain_2_loading
                self.remain_3 += self.remain_3_loading
                self.remain_1_loading = 0
                self.remain_2_loading = 0 
                self.remain_3_loading = 0 
                self.logger.info(f'TIME={self.time}: 装配台{self.id} 进料仓物料补充完成，现有 半成品1:{self.remain_1} ' \
                            f'半成品2:{self.remain_2} 半成品3:{self.remain_3}')
        if self.loading_time > 0:
            self.loading_time -= 1
            if self.loading_time == 0:
                self.logger.info(f'TIME={self.time}: 装配台{self.id} 出料仓装载AGV完成，现有成品{self.remain_finished_goods}个')

    def AGV_in(self, AGV_GLOBAL_ID, nums_list):
        num_1 = nums_list.get('1')
        num_2 = nums_list.get('2')
        num_3 = nums_list.get('3')
        num_tools = nums_list.get('tools')
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        flag = 0
        if not self.check_agv_in_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达装配台{self.id}进料区或AGV状态不在等待中, 卸载失败.')
            return False
        # 1
        if agv.remain_1 >= num_1:
            self.remain_1_loading += num_1
            agv.remain_1 -= num_1
            flag = 1
        else:
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 实际携带半成品1数量{agv.remain_1} 少于需要在装配台{self.id}卸载的数量{num_1}')
        # 2
        if agv.remain_2 >= num_2:
            self.remain_2_loading += num_2
            agv.remain_2 -= num_2
            flag = 1
        else:
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 实际携带半成品2数量{agv.remain_2} 少于需要在装配台{self.id}卸载的数量{num_2}')
        # 3
        if agv.remain_3 >= num_3:
            self.remain_3_loading += num_3
            agv.remain_3 -= num_3
            flag = 1
        else:
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 实际携带半成品3数量{agv.remain_3} 少于需要在装配台{self.id}卸载的数量{num_3}')
        # tools
        if agv.remain_tools > 0 and agv.remain_tools >= num_tools:
            self.tools_num += num_tools
            agv.remain_tools -= num_tools
            flag = 1
            self.logger.info(f'TIME={self.time}: 装配台{self.id}接收到来自AGV{agv.id}的工具{num_tools}个, 现有工具{self.tools_num}个')
        elif agv.remain_tools > 0 and agv.remain_tools < num_tools:
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 实际携带工具数量{agv.remain_tools} 少于需要在装配台{self.id}卸载的数量{num_tools}')
        if flag:
            self.unloading_time = AGV_UNLOAD_TIME
            agv.begin_unload()
        else:
            self.logger.warning(f'TIME={self.time}: AGV{agv.id} 未实际携带所需半成品，卸载失败')

    def AGV_out(self, AGV_GLOBAL_ID, nums_list):
        num = nums_list.get('goods')
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if not self.check_agv_out_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达装配台{self.id}出料区或AGV状态不在等待中, 装载失败.')
            return False
        if num > self.remain_finished_goods:
            self.logger.error(f'TIME={self.time}: 装配台{self.id}现有成品{self.remain_finished_goods} 小于 需要AGV{agv.id} 运走的数量{num}')
        else:
            agv.remain_goods += num
            self.remain_finished_goods -= num
        agv.begin_load()
        self.loading_time = AGV_LOAD_TIME
    
    @property
    def in_pos(self):
        return self.in_x, self.in_y
    
    @property
    def out_pos(self):
        return self.out_x, self.out_y
     

class FinishGoodsStorage:
    """成品存储仓"""
    def __init__(self, GLOBAL_ID, entity_manager, pos_x=900, pos_y=100, logger=None, control_center=None):
        self.GLOBAL_ID = GLOBAL_ID
        self.NAME = 'FinishGoodsStorage'
        self.entity_manager = entity_manager
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.remain_goods = 0  # 成品数目
        self.unloading_goods = 0 
        self.unloading_time = 0
        self.logger = logger
        self.time = 0
        self.control_center = control_center

    def step(self, time):
        self.time = time
        if self.unloading_time > 0:
            self.unloading_time -= 1
            if self.unloading_time == 0:
                self.remain_goods += self.unloading_goods
                self.logger.info(f'TIME={self.time}: 成品存储仓，成功接收成品{self.unloading_goods}个, 总计成品{self.remain_goods}个')
                self.control_center.agv_reward += 100 * self.unloading_goods  # 设置成品数目奖励，鼓励AGV完成更多的成品的生产 
                #self.control_center.select_reward += 100 * self.unloading_goods  # 设置成品数目奖励，鼓励AGV完成更多的成品的生产 
                # self.control_center.finishgoods_reward += self.unloading_goods
                self.unloading_goods = 0
                

    def check_agv_position(self, AGV_GLOBAL_ID):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if agv.position_x == self.pos_x and agv.position_y == self.pos_y and agv.state == 0:
            return True
        return False

    def AGV_in(self, AGV_GLOBAL_ID, num_list=[]):
        agv = self.entity_manager.get_entity_by_id(AGV_GLOBAL_ID)
        if not self.check_agv_position(AGV_GLOBAL_ID):
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未到达成品存储仓或AGV状态不在等待中,卸载失败.')
            return False
        if agv.remain_goods > 0:
            self.unloading_goods += agv.remain_goods
            
            # self.control_center.reward += 10.0 * agv.remain_goods

            self.logger.info(f'TIME={self.time}: AGV{agv.id} 开始向成品存储仓，卸载成品{agv.remain_goods}个')
            agv.begin_load()
            agv.remain_goods = 0
            self.unloading_time = AGV_UNLOAD_TIME
        else:
            self.logger.error(f'TIME={self.time}: AGV{agv.id} 未携带成品,卸载失败.')

    @property
    def in_pos(self):
        return self.pos_x, self.pos_y


if __name__ == '__main__':
    logger = getLogging()
    logger.info('================= A New Begining =================')
    # AGV_and_AssemblyStand_test_fuc()

    entityManager = EntityManager()
    orderManager = OrderManager()
    taskManager =  TaskManager()
    controlCenter = ControlCenter(0, entityManager, taskManager, orderManager, logger=logger)
    ORDER_ID += 1
    orderManager.create_order(ORDER_ID, earlist=100, lastest=500, products=2)
    ORDER_ID += 1
    orderManager.create_order(ORDER_ID, earlist=100, lastest=500, products=5)
    ORDER_ID += 1
    orderManager.create_order(ORDER_ID, earlist=100, lastest=500, products=2)
    print(orderManager.get_order_by_id(1))
    print(orderManager.get_order_by_id(2))
    print(orderManager.get_order_by_id(3))

    order = orderManager.get_order_by_id(1)
    # 生成第一阶段运输任务
    if order['state'] == 0:
        controlCenter.generate_task(1, order['products'])
