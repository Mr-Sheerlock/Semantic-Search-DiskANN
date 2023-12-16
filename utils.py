import random
import Graph as G
import numpy as np

# Get vertices from DB and insert them into the graph
def Initialize_Random_Graph(DBPath='DB1K.csv',R=5):
    DBGraph=G.Graph()
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
    return DBGraph

#gets euclidean distance between 2 vectors
def get_distance(v1,v2):
    return  np.linalg.norm(v1-v2)

#gets medoid of a graph
def get_medoid(DBGraph):
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




#gets min distance from any vertex in Anyset to Query
def get_min_dist (Anyset,Query):
    min_dist=10000000000000000000
    min_vertex=None
    for vertex in Anyset:
        dist=get_distance(vertex.value,Query)
        if(dist<min_dist):
            min_dist=dist
            min_vertex=vertex
    return min_vertex,min_dist

#initially, start is the medoid
# s is a vertex, Query is a vector
# k is a number, L is a number
def Greedy_Search(start,Query,k,L):
    search_List={start}
    Visited={}
    possible_frontier=search_List.difference(Visited)
    while possible_frontier != {}:
        p_star,_= get_distance(possible_frontier,Query)
        search_List.add(p_star.neighbors)
        Visited.add(p_star)
        if(len(search_List)>L):
            #update search list to retain closes L points to x_q
            search_List.sort(key=lambda x: get_distance(x.value,Query))
            search_List=search_List[:L]
        possible_frontier=search_List.difference(Visited)
    search_List.sort(key=lambda x: get_distance(x.value,Query))
    search_List=search_List[:k]
    return search_List,Visited
