import random
from Graph import Graph, Vertex
import numpy as np
import pandas as pd
import pickle
import os

class VecDB():
    #TODO: 
    # 2 Constructors:
        # VecDB() followed by db.insert_records(records_dict)
        # VecDB(file_path , new_db = False) 
        # where file path is the path to the binary file having the index
        # retrieve(query, top_k)
    # L R 
    tenKparams=(15,17)
    hundredKparams=(20,20)
    
    
    # file path of binary file 
    def __init__(self, file_path='DBIndex/', new_db=True):
        # R= 17, L= 15, alpha = 2, K= 5,
        self.RecordsPerCluster=10**4
        if (new_db):
            #TODO: Check the records per cluster here or at insert records
            self.L,self.R = VecDB.tenKparams
        else: 
            self.L,self.R = VecDB.hundredKparams  
        self.alpha = 2
        self.offset = 0
        self.IndexPath = file_path
        self.DBGraph=None
        self.currentfile=0
        if(not new_db):
            # load graph from binary file
            #TODO: remove hard coded value
            self.DBGraph= pickle.load(open(file_path+"0.bin", "rb"))
    
    
    
    
    #TODO: CHECK after klam el mo3eed: might need to handle enk t4oof el directory 
    # records are list of dictionaries containing id and embeddings
    def insert_records(self, records):
        # might wanna change records per cluster if records are more than 100k
        if(len(records)>self.RecordsPerCluster):
            # split into clusters of 10k
            for i in range(0, len(records), self.RecordsPerCluster):
                temp = records[i:(i + self.RecordsPerCluster)]
                #TODO: check this line
                self.insert_records(temp)
                self.currentfile+=1
            return
        
        self.DBGraph = self.Initialize_Random_Graph(records)
        self.Build_Index()
        
        # handle directory doesn't exist
        if not os.path.exists(self.IndexPath):
            os.makedirs(self.IndexPath)
        # set current file to len of current files in director
        self.currentfile+=len(os.listdir(self.IndexPath))
        with open (self.IndexPath+str(self.currentfile)+".bin", 'wb') as f:
            pickle.dump(self.DBGraph, f)
    #TODO:  CHECK return to this later
    
    def load_binary_data(self, binary_file_path):
    # Load the data from the binary file
        with open(binary_file_path, 'rb') as f:
            data = pickle.load(f)
        # offset of last inserted record
        self.offset = data.iloc[0][0]
        # print(data)
        return data

    # Initialize a Graph connected randomly from the given records
    # records is a list of dictionaries containing id and embeddings
    # the function also calls the graph to calculate its medoid
    def Initialize_Random_Graph(self, records):
        DBGraph=Graph()
        DBdata = [[x["id"]] + x["embed"] for x in records]
        # print(len(DBdata))
        for row in DBdata:
            dataKey = int(row[0])
            dataValue= np.array(row[1:],dtype=float)
            # normalize 1 time initially
            dataValue /= np.linalg.norm(dataValue)
            DBGraph.add_vertex(Vertex(dataKey, dataValue))

        size= len(DBGraph.verticies)
        # for vertex in DBGraph:
        #     print(vertex)
        # print(size)
        # add to offset for when writing the next cluster of vertices
        if(size==0 or size==1):
            return
        # first element in records
        # self.offset=DBdata[0]["id"]
        #Building Edges
        for vertex in DBGraph:
            # we want R neighbors
            for i in range(self.R):
                # neighbor= DBGraph.get_vertex(int(random.random()*size) + self.offset)
                neighbor= DBGraph.get_vertex(int(random.random()*size))
                while(neighbor==vertex):
                    # neighbor= DBGraph.get_vertex(int(random.random()*size) + self.offset)
                    neighbor= DBGraph.get_vertex(int(random.random()*size))
                DBGraph.add_edge((vertex.key,vertex.value),(neighbor.key,neighbor.value))
        # get medoid of graph ie. calculate it if it's not calculated
        DBGraph.get_Medoid()

        return DBGraph

    def index_to_distance(self,id, query):
            vertex = self.DBGraph.get_vertex(id)
            dist = self.get_distance(vertex.value, query)
            return dist
    
    def retrieve(self,query,k):
        query /= np.linalg.norm(query)

        # top k from all clusters
        ClustersResults = []
        for idx, filename in enumerate(os.listdir(self.IndexPath)):
            filepath = os.path.join(self.IndexPath, filename)
            
            self.DBGraph = pickle.load(open(filepath,"rb"))

            TopK,_= self.Greedy_Search(self.DBGraph.medoid,query, k)
            # distance,index=((self.index_to_distance(k, query), k+(idx* self.offset)) for k in TopK)
            # ListToAdd=
            # add top k from this cluster to the list
            # print(ListToAdd)
            ClustersResults.extend([(self.index_to_distance(VertexId, query), VertexId) for VertexId in TopK])
            self.offset+=len(self.DBGraph.verticies)
            del self.DBGraph

        # print(ClustersResults)
        # sorts on first element of the tuple (which are the distances)
        ClustersResults.sort()
        # print(ClustersResults)
        # only get ids
        ClustersResults = [element[1] for element in ClustersResults[:k]]
        return ClustersResults
    
    # gets euclidean distance between 2 vectors
    def get_distance(self,v1,v2):
        dist = np.dot(v1,v2) #/ (np.linalg.norm(v1) * np.linalg.norm(v2))
        # print(v1.shape, v2.shape)
        return 1.0 - dist
        # return  np.linalg.norm(v1-v2)


    def get_min_dist_Key (self,AnyKeysSet,Query):
        # arrKey = list(AnyKeysSet)
        # arrEmb = np.array([self.DBGraph.get_vertex(i).value for i in arrKey])

        # a_norm = np.linalg.norm(arrEmb, axis=1)
        # b_norm = np.linalg.norm(Query)
        # dist = (arrEmb @ Query) / (a_norm * b_norm)

        # # minDist = np.linalg.norm(arrEmb, axis=1)
        # minIndex = np.argmin(dist)
        # return arrKey[minIndex], -1

        # put to 50 because the max distance is 2 anyway 
        min_dist=50
        min_vertex=None
        for vertexKey in AnyKeysSet:
            # print("vertex",vertex)
            # print("Query",Query[:3])
            vertex=self.DBGraph.get_vertex(vertexKey)
            dist=self.get_distance(vertex.value,Query)
            # print(dist)
            if(dist<min_dist):
                min_dist=dist
                min_vertex=vertex.key
        return min_vertex,min_dist


    #initially, start is the medoid
    # s is a vertex, Query is a vector
    # k is a number, L is a number
    # TODO: change them to be indices instead of vertices to save ram
    def Greedy_Search(self,start,Query, k):
        search_List={start.key}
        Visited=set()
        #TODO: make the visited and the possible frontier set of indices instead of vertices to save ram.
        possible_frontier=search_List
        Query /= np.linalg.norm(Query)
        while possible_frontier != set():
            # print('possible_frontier',possible_frontier)
            p_star,_= self.get_min_dist_Key(possible_frontier,Query)

            # print('pstar',p_star)
            # if p_star==None:
            #     # break
            #     print('frontier: ')
            #     for v in possible_frontier:
            #         print(v)
            #     print(possible_frontier==set())
            search_List=search_List.union(self.DBGraph.get_vertex(p_star).neighbors)
            Visited.add(p_star)
            if(len(search_List)>self.L):
                #update search list to retain closes L points to x_q
                search_ListL_L=list(search_List)
                search_ListL_L.sort(key=lambda x: self.get_distance(self.DBGraph.get_vertex(x).value,Query))
                # only maintain L closest points
                search_ListL_L=search_ListL_L[:self.L]
                search_List=set(search_ListL_L)

            possible_frontier=search_List.difference(Visited)

        search_ListL_L=list(search_List)
        search_ListL_L.sort(key=lambda x: self.get_distance(self.DBGraph.get_vertex(x).value,Query))
        # only maintain k closest points
        search_ListL_L=search_ListL_L[:k]
        search_List=set(search_ListL_L)
        # both are vectors of integers
        return search_List,Visited

    # # Robust pruning
    #candidate set is set of integers
    def Robust_Prune(self,point,candidate_set,alpha):
        # print(candidate_set)
        candidate_set=candidate_set.union(point.neighbors)
        # candidate_set.difference({point.key}) # changed
        candidate_set=candidate_set.difference({point.key}) # changed
        point.neighbors=set()
        # print("candidate_set", candidate_set)
        # while candidate_set not empty
        while candidate_set:
            p_star,_= self.get_min_dist_Key(candidate_set,point.value)
            point.neighbors.add(p_star)
            if(len(point.neighbors)==self.R):
                break
            DummySet=candidate_set.copy()
            for candidatePointKey in candidate_set:
                candidatePoint=self.DBGraph.get_vertex(candidatePointKey)
                # print(alpha * self.get_distance(self.DBGraph.get_vertex(p_star).value,candidatePoint.value), " <= ", self.get_distance(candidatePoint.value,point.value))
                # print(alpha)
                # ">=" condition is reversed as we use cosine similarity (paper uses L2), so higher value means it is closer
                if(alpha * self.get_distance(self.DBGraph.get_vertex(p_star).value,candidatePoint.value) <= self.get_distance(candidatePoint.value,point.value)):
                    # print("HEEEEEEEEEEEEEEEEELPPPPPPPPPPP !!!!!!!!!!!!!!!!!!!!!!")
                    DummySet.remove(candidatePoint.key)
            candidate_set=DummySet


    def Build_Index(self):

        # R = min(R, len(dataset))
        self.iterationOverGraph(1) #alpha=1
        self.iterationOverGraph(self.alpha) #alpha=2


    def iterationOverGraph(self,alpha):
        # print('medoid',medoid)
        randIndex = list(self.DBGraph.get_vertices())
        random.shuffle(randIndex)
        # random permutation + sequential graph update
        for n in randIndex:
            node = self.DBGraph.get_vertex(n)
            # print(n)

            (_,V) = self.Greedy_Search(self.DBGraph.medoid, node.value, 1) #K=1
            
            self.Robust_Prune(node, V, alpha)
            
            neighbors = node.get_neighbors()

            for inbKey in neighbors:

                # CHECK : The backward edge is always added
                # check here in case we shouldn't add it in all cases ? Might be incorrect?
                inb=self.DBGraph.get_vertex(inbKey)
                if len(inb.get_neighbors()) > self.R:
                    # print("inb.get_neighbors()", inb.get_neighbors())
                    U = inb.get_neighbors().union({node.key})
                    # inb.add_neighbor(node.key)
                    self.Robust_Prune(inb, U, alpha)
                else:
                    inb.add_neighbor(node.key)