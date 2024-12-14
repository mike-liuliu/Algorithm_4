#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <limits>

using namespace std;

// Helper function to print a matrix
void printMatrix(const vector<vector<double>>& matrix, const string& name) {
    cout << name << ":" << endl;
    for (const auto& row : matrix) {
        for (double value : row) {
            cout << fixed << setprecision(6) << value << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Helper function to compare two matrices
bool areMatricesEqual(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    const double epsilon = 1e-6;
    int n = matrix1.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > epsilon) {
                cout << "Mismatch at (" << i << ", " << j << "): "
                     << "Matrix1 = " << matrix1[i][j] << ", Matrix2 = " << matrix2[i][j] << endl;
                return false;
            }
        }
    }
    return true;
}

// Class to construct MST using Prim's algorithm
class PrimMST {
public:
    int V;
    vector<vector<double>> graph;

    PrimMST(int vertices) : V(vertices), graph(vertices, vector<double>(vertices, 0)) {}

    vector<tuple<int, int, double>> constructMST() {
        vector<double> key(V, numeric_limits<double>::max());
        vector<int> parent(V, -1);
        vector<bool> mstSet(V, false);
        key[0] = 0;

        for (int count = 0; count < V - 1; count++) {
            int u = minKey(key, mstSet);
            mstSet[u] = true;

            for (int v = 0; v < V; v++) {
                if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v]) {
                    parent[v] = u;
                    key[v] = graph[u][v];
                }
            }
        }

        vector<tuple<int, int, double>> MST;
        for (int i = 1; i < V; i++) {
            MST.emplace_back(parent[i], i, graph[i][parent[i]]);
        }
        return MST;
    }

private:
    int minKey(const vector<double>& key, const vector<bool>& mstSet) {
        double min = numeric_limits<double>::max();
        int min_index = -1;

        for (int v = 0; v < V; v++) {
            if (!mstSet[v] && key[v] < min) {
                min = key[v];
                min_index = v;
            }
        }
        return min_index;
    }
};

// Create a random distance matrix
vector<vector<double>> createDistanceMatrix(int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, 1000);

    vector<vector<double>> distanceMatrix(N, vector<double>(N, 0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i < j) {
                double weight = dist(gen);
                distanceMatrix[i][j] = weight;
                distanceMatrix[j][i] = weight;
            }
        }
    }
    return distanceMatrix;
}

// MMJ matrix calculation using Floyd-Warshall variant
vector<vector<double>> calculateMMJMatrixFloydWarshall(const vector<vector<double>>& distanceMatrix) {
    int n = distanceMatrix.size();
    vector<vector<double>> mmjMatrix = distanceMatrix;

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mmjMatrix[i][j] = min(mmjMatrix[i][j], max(mmjMatrix[i][k], mmjMatrix[k][j]));
            }
        }
    }
    return mmjMatrix;
}

// MMJ matrix calculation using Algorithm 4 (Calculation and Copy)
vector<vector<double>> calculateMMJMatrixAlgo4(const vector<vector<double>>& distanceMatrix) {
    int n = distanceMatrix.size();
    vector<vector<double>> mmjMatrix(n, vector<double>(n, 0));

    // Construct MST
    PrimMST mst(n);
    mst.graph = distanceMatrix;
    auto mstEdges = mst.constructMST();

    // Sort edges by weight in descending order
    sort(mstEdges.begin(), mstEdges.end(), [](const auto& a, const auto& b) {
        return get<2>(a) > get<2>(b);
    });

    set<int> tree1Nodes, tree2Nodes;
    vector<set<int>> adjacencyList(n);
    for (const auto& edge : mstEdges) {
        adjacencyList[get<0>(edge)].insert(get<1>(edge));
        adjacencyList[get<1>(edge)].insert(get<0>(edge));
    }

    for (const auto& edge : mstEdges) {
        int u = get<0>(edge), v = get<1>(edge);
        double weight = get<2>(edge);

        adjacencyList[u].erase(v);
        adjacencyList[v].erase(u);

        vector<bool> visited(n, false);
        tree1Nodes.clear();
        tree2Nodes.clear();

        function<void(int, set<int>&)> dfs = [&](int node, set<int>& tree) {
            visited[node] = true;
            tree.insert(node);
            for (int neighbor : adjacencyList[node]) {
                if (!visited[neighbor]) dfs(neighbor, tree);
            }
        };

        dfs(u, tree1Nodes);
        dfs(v, tree2Nodes);

        for (int p1 : tree1Nodes) {
            for (int p2 : tree2Nodes) {
                mmjMatrix[p1][p2] = mmjMatrix[p2][p1] = weight;
            }
        }
    }

    return mmjMatrix;
}

int main() {
    int N = 10000;
    auto distanceMatrix = createDistanceMatrix(N);
    cout << N << endl;
    auto start = chrono::high_resolution_clock::now();
    auto mmjMatrixAlgo4 = calculateMMJMatrixAlgo4(distanceMatrix);
    auto end = chrono::high_resolution_clock::now();
    cout << "Time used (Algorithm 4): " << chrono::duration<double>(end - start).count() << " seconds" << endl;

    // int N = 100;
    // auto distanceMatrix = createDistanceMatrix(N);
    // cout << N << endl;    
    // auto start = chrono::high_resolution_clock::now();
    // auto mmjMatrixFloydWarshall = calculateMMJMatrixFloydWarshall(distanceMatrix);
    // auto end = chrono::high_resolution_clock::now();
 
    // cout << "Time used (Floyd-Warshall): " << chrono::duration<double>(end - start).count() << " seconds" << endl;

    // start = chrono::high_resolution_clock::now();
    // auto mmjMatrixAlgo4 = calculateMMJMatrixAlgo4(distanceMatrix);
    // end = chrono::high_resolution_clock::now();
    // cout << "Time used (Algorithm 4): " << chrono::duration<double>(end - start).count() << " seconds" << endl;

    // // Compare the two matrices
    // if (areMatricesEqual(mmjMatrixFloydWarshall, mmjMatrixAlgo4)) {
    //     cout << "The matrices are equal!" << endl;
    // } else {
    //     cout << "The matrices are NOT equal!" << endl;
    // }

    return 0;
}
