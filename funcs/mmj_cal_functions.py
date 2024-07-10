import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from ctypes import cdll, c_char_p
import numpy.ctypeslib as ctl
from sklearn.metrics import pairwise_distances
import ctypes
 
 

# The cal_mmj_matrix_by_shortest_path_on_minimum_spanning_tree function is based on Madhav-99's code. see:
# https://github.com/Madhav-99/Minimax-Distance

def cal_mmj_matrix_by_shortest_path_on_minimum_spanning_tree(X):
    import networkx as nx
    lenX = len(X)
    round_n = 15
    distance_matrix = pairwise_distances(X)
    distance_matrix = np.round(distance_matrix,round_n)
    G = nx.Graph()
    for i in range(lenX):
        G.add_node(i)   

    for i in range(lenX):
        for j in range(lenX):
            if(i!=j):
                G.add_edge(i,j,weight=distance_matrix[i][j])
    MST = nx.minimum_spanning_tree(G)

    mmj_matrix = np.zeros((lenX, lenX))
    
    for i in range(lenX):
        for j in range(lenX):
            if j > i:
                max_weight = -1 
                path = nx.shortest_path(MST, source=i, target=j)
                for k in range(len(path)-1):
                    if( MST.edges[path[k],path[k+1]]['weight'] > max_weight):
                        max_weight = MST.edges[path[k],path[k+1]]['weight']
                mmj_matrix[i,j] = mmj_matrix[j,i] = max_weight
 
    return mmj_matrix


def construct_MST_from_graph(distance_matrix):
    import networkx as nx
    lenX = len(distance_matrix)
    g = construct_MST_prim(lenX)
    g.graph = distance_matrix
    MST_list = g.primMST()    
 
    MST = nx.Graph()
    for i in range(lenX):
        MST.add_node(i)
    for edge in MST_list:
        MST.add_edge(edge[0],edge[1],weight=edge[2])          
    return MST


def cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X, round_n = 15):
    import networkx as nx
    
    lenX = len(X)
    distance_matrix = pairwise_distances(X)
    distance_matrix = np.round(distance_matrix,round_n)
    mmj_matrix = np.zeros((lenX,lenX))

    MST = construct_MST_from_graph(distance_matrix)
    
    MST_edge_list = list(MST.edges(data='weight'))
 
    edge_node_list = [(edge[0],edge[1]) for edge in MST_edge_list]
    edge_weight_list = [edge[2] for edge in MST_edge_list]
    edge_large_to_small_arg = np.argsort(edge_weight_list)[::-1]
    edge_weight_large_to_small = np.sort(edge_weight_list)[::-1]
    edge_nodes_large_to_small = [edge_node_list[i] for i in edge_large_to_small_arg]
 
    for i, edge_nodes in enumerate(edge_nodes_large_to_small):
        edge_weight = edge_weight_large_to_small[i]
        MST.remove_edge(*edge_nodes)
        tree1_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[0]))
        tree2_nodes = list(nx.dfs_preorder_nodes(MST, source=edge_nodes[1]))
        for p1 in tree1_nodes:
            for p2 in tree2_nodes:
                mmj_matrix[p1, p2] = mmj_matrix[p2, p1] = edge_weight      
 
    return mmj_matrix

def cal_mmj_matrix_algo_1_python(X, round_n = 15):
    
    distance_matrix = pairwise_distances(X)
    distance_matrix = np.round(distance_matrix, round_n)

    lenX = len(X)
   
    mmj_matrix = np.zeros((lenX,lenX))

    mmj_matrix[0,1] = distance_matrix[0,1]
    mmj_matrix[1,0] = distance_matrix[1,0]
 
    for kk in range(2,lenX):
        cal_n_mmj(distance_matrix, mmj_matrix, kk)
    return mmj_matrix

def cal_mmj_matrix_algo_1_cpp(X, round_n = 15):
    distance_matrix = pairwise_distances(X)
 
    distance_matrix = np.round(distance_matrix,round_n)
    directory = os.getcwd()
    directory += "/mmj_so/cal_mmj_distance_matrix_macOS.so"
    lib = cdll.LoadLibrary(directory)
    mmj_matrix = np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.float64)
    py_cal_mmj_matrix = lib.cal_mmj_matrix
    py_cal_mmj_matrix.argtypes = [ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctypes.c_int] 
    py_cal_mmj_matrix(distance_matrix, mmj_matrix, len(distance_matrix)) 
    return mmj_matrix

 

def mmj_n_to_r(distance_matrix, mmj_matrix, n, r):
    max_jump_list = []
    for ttt in range(n):
        m_jump = np.max((distance_matrix[n,ttt],mmj_matrix[ttt,r]))
        max_jump_list.append(m_jump)
    return np.min(max_jump_list)

def mmj_r_to_n(distance_matrix, mmj_matrix, n, r):
    max_jump_list = []
    for ttt in range(n):
        m_jump = np.max((mmj_matrix[r,ttt], distance_matrix[ttt, n]))
        max_jump_list.append(m_jump)
    return np.min(max_jump_list)

     
def cal_n_mmj(distance_matrix, mmj_matrix, n):
    for i in range(n):
        mmj_matrix[n,i] = mmj_n_to_r(distance_matrix, mmj_matrix, n, i)
        mmj_matrix[i,n] = mmj_r_to_n(distance_matrix, mmj_matrix, n, i)
        
    for i in range(n):        
        for j in range(n):
            if i < j:
                mmj_matrix[i,j] =  update_mmj_ij(distance_matrix, mmj_matrix, n, i, j)
                mmj_matrix[j,i] =  update_mmj_ij(distance_matrix, mmj_matrix, n, j, i)
                
def update_mmj_ij(distance_matrix, mmj_matrix, n, i,j):
    m1 = mmj_matrix[i,j]
    m2 = np.max((mmj_matrix[i,n],mmj_matrix[n,j]))
    return np.min((m1,m2))

def cal_Widest_path_problem_matrix_by_algo_4(pairwise_bandwidth_matrix):
    import networkx as nx
    N = len(pairwise_bandwidth_matrix)
    
    graph = pairwise_bandwidth_matrix

    MST_list = maximumSpanningTree(graph)  

    Widest_path_matrix = np.zeros((N,N))
 
    MST = nx.Graph()
    for i in range(N):
        MST.add_node(i)
    for yy in MST_list:
        MST.add_edge(yy[0],yy[1],weight=yy[2])
 
    edge_node_list = [(i[0],i[1]) for i in MST_list]
    edge_len_list = [i[2] for i in MST_list]
    edge_small_to_large_arg = np.argsort(edge_len_list)
    edge_small_to_large = np.sort(edge_len_list)
    new_edge_node_list = [edge_node_list[i] for i in edge_small_to_large_arg]

    for i, e in enumerate(new_edge_node_list):
        edge_ll = edge_small_to_large[i]
        MST.remove_edge(*e)
        tree1_nodes = list(nx.dfs_preorder_nodes(MST, source=e[0]))
        tree2_nodes = list(nx.dfs_preorder_nodes(MST, source=e[1]))
        for p1 in tree1_nodes:
            for p2 in tree2_nodes:
                Widest_path_matrix[p1, p2] = Widest_path_matrix[p2, p1] = edge_ll      
 
    return Widest_path_matrix