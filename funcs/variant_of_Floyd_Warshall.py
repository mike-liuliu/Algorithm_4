
# See:
# https://www.naukri.com/code360/library/floyd-warshall-algorithm-at-a-glance
# https://shadyf.com/blog/notes/2016-09-20-APSP-problem/

def cal_mmj_variant_of_Floyd_Warshall_python(X, round_n = 15):
    distance_matrix = pairwise_distances(X).round(round_n) 
    n = len(X)    
    p = distance_matrix.copy()
 
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if i != k and j != k:
                        p[j,k] = min (p[j,k], max (p[j,i], p[i,k])) 
    return p

def cal_mmj_matrix_Floyd_Warshall_cpp(X, round_n = 15):
    distance_matrix = pairwise_distances(X).round(round_n)
    directory = os.getcwd()
    directory += "/mmj_so/cal_mmj_Floyd_Warshall_macOS.so"
    lib = cdll.LoadLibrary(directory)
    mmj_matrix = np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.float64)
    py_cal_mmj_matrix = lib.cal_mmj_matrix
    py_cal_mmj_matrix.argtypes = [ctl.ndpointer(np.float64), ctl.ndpointer(np.float64), ctypes.c_int] 
    py_cal_mmj_matrix(distance_matrix, mmj_matrix, len(distance_matrix)) 
    return mmj_matrix