class Vertex(object):
    def __init__(self, key):
        self.key = key
        self.neighbors = {}
        self.value=[None]

    def __init__(self, key,value):
        self.key = key
        self.neighbors = {}
        self.value=value
    def add_neighbor(self, neighbor, weight=0):
        self.neighbors[neighbor] = weight

    def __str__(self):
        return '{} neighbors: {} \n values: {}'.format(
            self.key,
            [x.key for x in self.neighbors]
            ,self.value[:3]
        )

    def get_connections(self):
        return self.neighbors.keys()

    def get_weight(self, neighbor):
        return self.neighbors[neighbor]
    
class Graph(object):
    def __init__(self):
        self.verticies = {}

    def add_vertex(self, vertex):
        self.verticies[vertex.key] = vertex

    def get_vertex(self, key):
        try:
            return self.verticies[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return key in self.verticies

    def add_edge(self, v1_info, v2_info, weight=0):
        v1_key,v1_value=v1_info
        v2_key,v2_value=v2_info
        if v1_key not in self.verticies:
            self.add_vertex(Vertex(v1_key,v1_value))
        if v2_key not in self.verticies:
            self.add_vertex(Vertex(v2_key,v2_value))
        self.verticies[v1_key].add_neighbor(self.verticies[v2_key], weight)
        self.verticies[v2_key].add_neighbor(self.verticies[v1_key], weight)

    def get_vertices(self):
        return self.verticies.keys()

    def __iter__(self):
        return iter(self.verticies.values())