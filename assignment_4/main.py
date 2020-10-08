import torch
import numpy
import os
import matplotlib
from datetime import datetime, timedelta

def strip_lines(lines):
    lines = [line.strip() for line in lines]
    lines = [float(line.split("   ")[1]) for line in lines]
    return lines
    
# get all files in directory
files_abnormal = os.listdir("ecg/normal")

# each file comes in three pairs:
# .0    .1    and  .ann
# we want only the filename
files = set()

for file in files_abnormal:
    files.add(file.rsplit(".", 1)[0])
    
files = list(files)

# '801_000035' -> ['801', '000035']
files.sort(key=lambda x: x.split("_")[1])

timestamps = []
lead_0 = []
lead_1 = []

for file in files:
    with open(f"ecg/normal/{file}.0", "r") as f:
        data_0 = strip_lines(f.readlines())
    
    with open(f"ecg/normal/{file}.1", "r") as f:
        data_1 = strip_lines(f.readlines())
    
    with open(f"ecg/normal/{file}.ann", "r") as f:
        data_ann = f.readlines()
        data_ann = [line.strip() for line in data_ann]
        data_ann = [line.split(" ")[0] for line in data_ann]

    lead_0.append(data_0)
    lead_1.append(data_1)
    timestamps.append(datetime.strptime(data_ann[0], '%M:%S.%f'))
    
x_values = []
y_values = []

start = datetime.now()

for i, row in enumerate(lead_0[0]):
    x_values.append(start + timedelta(seconds=4) * i)
    y_values.append(row[i])

# for i in range(0, len(timestamps) - 1, 2):
#     # start = timestamps[i] 
#     # end = timestamps[i + 1]
#     # delta = end - start
#     # total_seconds = delta.total_seconds()
#     # interval = total_seconds / len(lead_0[i])
#     # data_points = len(lead_0[i])
#     # print("Start:", start)
#     # print("End:", end)
#     # print("Seconds:", delta.total_seconds())
#     #print("Interval:", interval)
#     #print("Data points:", data_points)
#     #print("\n")
#     #print("Start type:", type(start))
    
#     for j in range(len(lead_0[i])):
#         x_values.append(start + timedelta(seconds=4) * j * i)
#         y_values.append(lead_0[i][j])
        
import matplotlib.pyplot as plt

plt.plot(x_values, y_values)
plt.ylim((800, -800))
plt.show()