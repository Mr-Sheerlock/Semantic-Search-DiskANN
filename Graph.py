import numpy as np
class Vertex(object):
    def __init__(self, key):
        self.key = key # 4 bytes
        self.neighbors = set()
        self.value=[None]

    def __init__(self, key,value):
        self.key = key
        self.neighbors = set()
        self.value=value

    def add_neighbor(self, neighbor):
        self.neighbors.add(neighbor)

    def __str__(self):
        return '{} neighbors: {} \n values: {}'.format(
            self.key,
            [x for x in self.neighbors]
            ,self.value[:3]
        )
    # get neighbor vertices
    def get_neighbors(self):
        return self.neighbors

class Graph(object):
    def __init__(self):
        self.verticies = {}
        self.medoid=None
    
    def add_vertex(self, vertex):
        self.verticies[vertex.key] = vertex

    def get_vertex(self, key):
        try:
            return self.verticies[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return key in self.verticies

    def add_edge(self, v1_info, v2_info):
        v1_key,v1_value=v1_info
        v2_key,v2_value=v2_info
        if v1_key not in self.verticies:
            self.add_vertex(Vertex(v1_key,v1_value))
        if v2_key not in self.verticies:
            self.add_vertex(Vertex(v2_key,v2_value))
        self.verticies[v1_key].add_neighbor(v2_key)

    def get_vertices(self):
        return self.verticies.keys()
    
    # calculates medoid of graph
    def get_Medoid(self,offset):
        if self.medoid is None:
            vX = list(self.get_vertices())
            Embeddings = [self.verticies[i].value for i in vX]
            vMean = np.mean(Embeddings, axis=0)                               # compute centroid
            minIndex=np.argmin([sum((x - vMean)**2) for x in Embeddings])
            medoid = self.get_vertex(minIndex+offset)
            self.medoid=medoid
        return self.medoid
    
    
    

    def __iter__(self):
        return iter(self.verticies.values())

    def __str__(self):
        return 'Graph({})'.format(dict(self.verticies))