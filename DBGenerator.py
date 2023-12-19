import numpy as np
import csv

rng = np.random.default_rng(50)

Db_size_power= 5
v=rng.random(((10**Db_size_power), 70), dtype=np.float32)
print(v.shape)
num_rows = v.shape[0]

row_indices = np.arange(num_rows).reshape((-1, 1))
arr_with_indices = np.hstack((row_indices, v))
# print(arr_with_indices)
# print(arr_with_indices.shape)
DB_Path="dataset/"
with open(DB_Path+'DB10K.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)


    for row in arr_with_indices:
        formatted_row = ['{:.8f}'.format(value) for value in row]
        csv_writer.writerow(formatted_row)


