import statistics
import numpy as np

with open("knn.txt") as file_in:
    lines = []
    for line in file_in:
        lines.append(float(line.strip("\n")))

data = np.array(lines)

print("Standard Deviation of the sample is % s "% (statistics.stdev(data)))
print("Mean of the sample is % s " % (statistics.mean(data)))