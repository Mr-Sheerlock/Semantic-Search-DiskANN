import numpy as np
import csv
import pandas as pd


rng = np.random.default_rng(50)

Db_size_power= 5
# Offset=0
first_time=True
Offset=0
if not first_time:
    Offset=10**3 #3adad el 7agat ele katabtaha fel file 2abl keda
    v=rng.random(((10**Db_size_power), 70), dtype=np.float32)
    del v

v=rng.random(((10**Db_size_power), 70), dtype=np.float32)
print(v.shape)
num_rows = v.shape[0]

row_indices = np.arange(num_rows).reshape((-1, 1))+Offset
arr_with_indices = np.hstack((row_indices, v))
# print(arr_with_indices)
# print(arr_with_indices.shape)
DB_Path="loler/"
with open(DB_Path+'DB100K.csv', 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)


    for row in arr_with_indices:
        formatted_row = ['{:.12f}'.format(value) for value in row]
        csv_writer.writerow(formatted_row)


# cities = pd.DataFrame(arr_with_indices,columns=None)
# cities.to_csv(DB_Path+'cities.csv',header=None,index=False)

