# this script aims to extract the positive ratio value in the log file.

log_file_1 = open("C:\\Users\\Zhuoming Liu\\Desktop\\slurm-10300314.out", "r")
log_file_2 = open("C:\\Users\\Zhuoming Liu\\Desktop\\slurm-10300313.out", "r")

def obtain_the_ratio(lines):
    all_pos_ratio = []
    for line in lines:
        if line.startswith('pos_ratio'):
            while line.startswith('pos_ratio'):
                line = line.strip('pos_ratio')
            line = line.strip()
            while line.startswith('pos_ratio'):
                line = line.strip('pos_ratio')
            try:
                value = float(line)
                all_pos_ratio.append(value)
            except:
                print(line)
                continue
    return all_pos_ratio
    
content_list_1 = log_file_1.readlines()
ratio_values_1 = obtain_the_ratio(content_list_1)
content_list_2 = log_file_2.readlines()
ratio_values_2 = obtain_the_ratio(content_list_2)

print('ratio_values_1', len(ratio_values_1))
print('ratio_values_2', len(ratio_values_2))
min_len = min(len(ratio_values_1), len(ratio_values_2))
ratio_values_1 = ratio_values_1[:min_len]
ratio_values_2 = ratio_values_2[:min_len]

x = [i*100 for i in range(min_len)]

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.plot(x, ratio_values_1 , color='red', label='trained base48 evaluated on base48')
plt.plot(x, ratio_values_2 , color='green', label='trained all80 evaluated on base48')

font1 = {'size': 10}
plt.legend(loc=4, prop=font1) # 显示图例
plt.grid()
plt.xlabel('recall', fontsize=13)
plt.ylabel('precision', fontsize=13)

plt.show()