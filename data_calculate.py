import pandas as pd

# 读取CSV文件
file_path = r'results/data.csv'
df = pd.read_csv(file_path)

# 计算平均值
finish_goods_avg = df['finish_goods_list'].mean()
order_intime_avg = df['order_intime_list'].mean()
total_task_mean_time_avg = df['total_task_mean_time'].mean()

# 打印结果
print(f"Finish Goods List Average: {finish_goods_avg}")
print(f"Order Intime List Average: {order_intime_avg}")
print(f"Total Task Mean Time Average: {total_task_mean_time_avg}")