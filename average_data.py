import numpy as np

num_to_average = 10
file_name = "dynamic day 5 60 to 120.csv"

data = np.genfromtxt(file_name, skip_header = True, delimiter = ',')[7:,:]
ave_data = []

i = 0
while (i < len(data) - 1):
    average = 0
    for j in range(num_to_average):
        if i == 0:
            i += 1
            continue
        average += data[i][1]
        if (i == len(data) - 1):
            break;
        i += 1
    average /= num_to_average
    ave_data.append(average)

np.savetxt("average_data.csv", ave_data, delimiter = ',')
