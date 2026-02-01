import csv

from matplotlib import pyplot as plt
import numpy as np

def running_mean(data, N=40):
    kernel = np.ones(N)
    conv_len = len(data) - N + 1
    mean_vals = np.zeros(conv_len)
    std_vals = np.zeros(conv_len)
    
    for i in range(conv_len):
        window = data[i:i + N]
        mean_vals[i] = np.mean(window)
        std_vals[i] = np.std(window)
    
    return mean_vals, std_vals

def read_csv_column(csv_file, column_index):
    column_data = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) > column_index:  # Ensure the row has enough columns
                column_data.append(float(row[column_index]))
    return column_data

# Paths to your CSV files
csv_files = ['data_d&s_new.csv']
labels = ['Dense to Sparse reward']

# Columns you want to plot (excluding Index)
columns_to_plot = [1, 2, 3, 4, 5]
column_labels = ['select_return_list', 'agv_return_list', 'finish_goods_list', 'order_intime_list', 'total_task_mean_time']

for col_idx, col_label in zip(columns_to_plot, column_labels):
    plt.figure(figsize=(6, 6))
    for file_idx, label in enumerate(labels):
        data = read_csv_column(csv_files[file_idx], col_idx)
        mean_vals, std_vals = running_mean(np.array(data), N=60)
        if file_idx == 1:
            mean_vals -=1
        # Plot smoothed data as line and get the line object
        line, = plt.plot(mean_vals, label=f'{label}')
        
        # Add shaded area for standard deviation (using 0.5 * std)
        plt.fill_between(range(len(mean_vals)), 
                         mean_vals - 0.5 * std_vals, 
                         mean_vals + 0.5 * std_vals, 
                         color=line.get_color(), alpha=0.1, )

    plt.title(col_label)
    plt.xlabel('Index')
    plt.ylabel(col_label)
    plt.xlim(-10, 250)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{col_label}-DS-D-S.png')
    plt.close()