import numpy as np
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
import pickle

DB_SEED_NUMBER=50

# rng = np.random.default_rng(DB_SEED_NUMBER)
# vectors = rng.random((20*10**6, 70), dtype=np.float32)
# for i in range(0,20_000_000, 5_000_000):
#   np.save(open(f"./Semantic-Search-DiskANN/DATA/vec{str(i//5_000_000)}.bin",'wb'), vectors[i:i+5_000_000])

# data = np.load(open("./Semantic-Search-DiskANN/DATA/vec2.bin","rb"))
# vec = data[0]
# vec = vec/np.linalg.norm(vec)
# print(vec[:3])

vectors = np.load(open("./Semantic-Search-DiskANN/DATA/vec2.bin","rb"))

db = VecDB(file_path='DBIndex5mil_2/')
# db.L,db.R= VecDB.hundredKparams
db.RecordsPerCluster=10**4
print(db.RecordsPerCluster)
print(db.L,db.R)
print("Inserting", len(vectors), "records")
records_dict = [{"id": (i+15_000_000), "embed": list(row)} for i, row in enumerate(vectors)]
db.insert_records(records_dict)