# Prim's Minimum Spanning Tree (MST) algorithm. 
# Based on the code from geeksforgeeks.org. See:
# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/

 
import sys
class construct_MST_prim():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = None

    # A utility function to print 
    # the constructed MST stored in parent[]
    def printMST(self, parent):
#         print("Edge \tWeight")
        
        MST = []
        for i in range(1, self.V):
#             print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
            MST.append([parent[i],i, self.graph[i][parent[i]]])
        return MST

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False \
                and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        MST_list = self.printMST(parent)
        
        return MST_list


# Calculate maximum-spanning-tree, See:
# https://www.geeksforgeeks.org/maximum-spanning-tree-using-prims-algorithm/
  
import sys
 
def findMaxVertex(visited, weights, V):
 
    index = -1
    maxW = -sys.maxsize
 
    for i in range(V): 
        if (visited[i] == False and weights[i] > maxW):
            maxW = weights[i]
            index = i
    return index
 
def printMaximumSpanningTree(graph, parent):
    V = len(graph)
    MST = []
    for i in range(1, V):
        MST.append([parent[i],i, graph[i][parent[i]]])
    return MST
 
def maximumSpanningTree(graph):

    V = len(graph)
    visited = [True]*V
    weights = [0]*V
    parent = [0]*V
    for i in range(V):
        visited[i] = False
        weights[i] = -sys.maxsize
 
    weights[0] = sys.maxsize
    parent[0] = -1
    for i in range(V - 1):
        maxVertexIndex = findMaxVertex(visited, weights, V)
        visited[maxVertexIndex] = True
        for j in range(V):
            if (graph[j][maxVertexIndex] != 0 and visited[j] == False):
                if (graph[j][maxVertexIndex] > weights[j]):
                    weights[j] = graph[j][maxVertexIndex]
                    parent[j] = maxVertexIndex
    MST_list = printMaximumSpanningTree(graph, parent)
    
    return MST_list

 
 
