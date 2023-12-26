import os
import pickle
def AddOffset(DBGraph, offset ):
    for i in DBGraph.get_vertices():
        DBGraph.verticies[i].key+=offset
    return DBGraph

for i in os.listdir("/path"):
    Graph=pickle.load(open("/path/"+i,"rb"))
    Graph=AddOffset(Graph,offset)