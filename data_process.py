import csv

from matplotlib import pyplot as plt
import numpy as np

def running_mean(x, N=60):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def read_csv_column(csv_file, column_index):
    column_data = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index:  # 确保该行有足够的列数
                column_data.append(row[column_index])
    return column_data

# 示例用法
csv_file = r'results/data_d.csv'  # 你的CSV文件路径
column_index = 1  # 你想要读取的列的索引
column_data = read_csv_column(csv_file, column_index)
column_data.pop(0)

# 将数据转换为浮点数
column_data = [float(x) for x in column_data]
print(column_data)
print(type(column_data[0]))
column_data = np.array(column_data)
column_data = running_mean(column_data)
plt.plot(column_data)
plt.title('select_return')
plt.savefig('select_return_processed_data.png')
plt.close()

# 示例用法
csv_file = r'results/data_d.csv'  # 你的CSV文件路径
column_index = 2  # 你想要读取的列的索引
column_data = read_csv_column(csv_file, column_index)
column_data.pop(0)

# 将数据转换为浮点数
column_data = [float(x) for x in column_data]
print(column_data)
print(type(column_data[0]))
column_data = np.array(column_data)
column_data = running_mean(column_data)
plt.plot(column_data)
plt.title('agv_return')
plt.savefig('agv_return_processed_data.png')
plt.close()

csv_file = r'results/data.csv'  # 你的CSV文件路径
column_index = 3    # 你想要读取的列的索引
column_data = read_csv_column(csv_file, column_index)
column_data.pop(0)

# 将数据转换为浮点数
column_data = [float(x) for x in column_data]
print(column_data)
print(type(column_data[0]))
column_data = np.array(column_data)
column_data = running_mean(column_data)
plt.plot(column_data)
plt.title('finish_goods_num')
plt.savefig('goods_processed_data.png')
plt.close()

csv_file = r'results/data.csv'  # 你的CSV文件路径
column_index = 4    # 你想要读取的列的索引
column_data = read_csv_column(csv_file, column_index)
column_data.pop(0)

# 将数据转换为浮点数
column_data = [float(x) for x in column_data]
print(column_data)
print(type(column_data[0]))
column_data = np.array(column_data)
column_data = running_mean(column_data)
plt.plot(column_data)
plt.title('order_intime_list')
plt.savefig('order_intime_list_process_data.png')
plt.close()

csv_file = r'results/data.csv'  # 你的CSV文件路径
column_index = 5    # 你想要读取的列的索引
column_data = read_csv_column(csv_file, column_index)
column_data.pop(0)

# 将数据转换为浮点数
column_data = [float(x) for x in column_data]
print(column_data)
print(type(column_data[0]))
column_data = np.array(column_data)
column_data = running_mean(column_data)
plt.plot(column_data)
plt.title('total_task_mean_time')
plt.savefig('total_task_mean_time_processed_data.png')
plt.close()