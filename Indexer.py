import random
import Graph as G
import numpy as np
DBGraph=None
R=5
# Get vertices from DB and insert them into the graph
def Initialize_Random_Graph():
    DBGraph=G.Graph()
    DBPath='DB1K.csv'
    with open(DBPath, 'r') as f:
        DB = f.readlines()
        DB = [x.strip() for x in DB]
        for i,x in enumerate(DB):
            x = x.split(',')
            x= np.array(x,dtype=float)
            DBGraph.add_vertex(G.Vertex(int(x[0]),x[1:]))
            # Graph.Insert(x[0], x[1], x[2])
        # Choose R random neighbors for each vertex
        size= len(DBGraph.verticies)
        if(size==0 or size==1):
            return
        for vertex in DBGraph:
            for i in range(R):
                neighbor= DBGraph.get_vertex(int(random.random()*size))
                while(neighbor==vertex):
                    neighbor= DBGraph.get_vertex(int(random.random()*size))
                DBGraph.add_edge((vertex.key,vertex.value),(neighbor.key,neighbor.value),0)
                DBGraph.add_edge((neighbor.key,neighbor.value),(vertex.key,vertex.value),0)
        return DBGraph
def get_distance(v1,v2):
    return  np.linalg.norm(v1-v2)
def get_medoid():
    min_distance=10000000000000000000
    medoid=None
    for vertex in DBGraph:
        current_total_distance=0
        for vertex2 in DBGraph:
            if(vertex==vertex2):
                continue
            dist=get_distance(vertex.value,vertex2.value)
            current_total_distance+=dist
        
        if(current_total_distance<min_distance):
            min_distance=current_total_distance
            medoid=vertex
    return medoid,min_distance


DBGraph=Initialize_Random_Graph()
for i,vertex in enumerate(DBGraph):
            print(vertex)
            if(i==3):
                break

print(get_medoid())