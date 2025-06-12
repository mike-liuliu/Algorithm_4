#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <set>
#include <queue>
#include <random>
#include <numeric>
#include <stack>
#include <chrono>
#include <limits>
#include <tuple>
#include <iomanip>
#include <functional>

using namespace std;

using Matrix = vector<vector<double>>;
using Edge = tuple<int, int, double>;
const double INF = numeric_limits<double>::infinity();

// Prim's MST with min-heap optimization
vector<int> primMST(const Matrix& dist) {
    int V = dist.size();
    vector<double> key(V, INF);
    vector<int> parent(V, -1);
    vector<bool> inMST(V, false);

    key[0] = 0.0;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    pq.emplace(0.0, 0);

    while (!pq.empty()) {
        auto [k, u] = pq.top(); pq.pop();
        if (inMST[u]) continue;
        inMST[u] = true;

        for (int v = 0; v < V; ++v) {
            if (dist[u][v] && !inMST[v] && dist[u][v] < key[v]) {
                key[v] = dist[u][v];
                parent[v] = u;
                pq.emplace(key[v], v);
            }
        }
    }
    return parent;
}

vector<set<int>> buildMSTGraph(int n, const vector<int>& parent) {
    vector<set<int>> mst(n);
    for (int i = 1; i < n; ++i) {
        mst[i].insert(parent[i]);
        mst[parent[i]].insert(i);
    }
    return mst;
}

void dfs(int start, const vector<set<int>>& graph, vector<bool>& visited, vector<int>& nodes) {
    stack<int> s;
    s.push(start);
    visited[start] = true;

    while (!s.empty()) {
        int node = s.top(); s.pop();
        nodes.push_back(node);
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                s.push(neighbor);
            }
        }
    }
}

vector<vector<double>> createDistanceMatrix(int N, int seed) {
    mt19937 gen(seed);
    uniform_int_distribution<> dist(1, 999);
    vector<vector<double>> A(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i][j] = dist(gen);

    vector<vector<double>> sym_A(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sym_A[i][j] = floor((A[i][j] + A[j][i]) / 2.0);

    for (int i = 0; i < N; ++i)
        sym_A[i][i] = 0.0;
    return sym_A;
}

void worker(int tid,
            const vector<set<int>>& base_mst,
            const vector<pair<int, int>>& edge_nodes_large_to_small,
            const vector<double>& edge_weights,
            Matrix& mmj_matrix,
            queue<int>& task_queue,
            mutex& queue_mutex) {

    auto MST_temp = base_mst;
    int current_removed = -1;
    int n = mmj_matrix.size();
    vector<bool> visited(n, false);

    vector<int> tree1, tree2;
    int task = -1;

    while (true) {

        {
            lock_guard<mutex> lock(queue_mutex);
            if (task_queue.empty()) return;
            task = task_queue.front();
            task_queue.pop();
        }

        for (int i = current_removed + 1; i <= task; ++i) {
            auto [u, v] = edge_nodes_large_to_small[i];
            MST_temp[u].erase(v);
            MST_temp[v].erase(u);
        }
        current_removed = task;

        auto [u, v] = edge_nodes_large_to_small[task];
        double weight = edge_weights[task];

        tree1.clear(); tree2.clear();
        dfs(u, MST_temp, visited, tree1);
        dfs(v, MST_temp, visited, tree2);

        for (int a : tree1)
            for (int b : tree2)
                mmj_matrix[a][b] = mmj_matrix[b][a] = weight;

        if (tree1.size() > 100 || tree2.size() > 100)
            fill(visited.begin(), visited.end(), false);
        else{
            for (int x : tree1) visited[x] = false;
            for (int x : tree2) visited[x] = false;
        }

    }
}

Matrix cal_mmj_matrix_by_algo_4_Calculation_and_Copy_parallel_compu(const Matrix& distance_matrix, int n_jobs) {
    int n = distance_matrix.size();
    Matrix mmj_matrix(n, vector<double>(n, 0.0));
    queue<int> task_queue;
    mutex queue_mutex;
 

    for (int i = 0; i < n - 1; ++i)
        task_queue.push(i);


    auto parent = primMST(distance_matrix);
    auto base_mst = buildMSTGraph(n, parent);

    vector<Edge> edge_list;
    for (int i = 1; i < n; ++i)
        edge_list.emplace_back(min(i, parent[i]), max(i, parent[i]), distance_matrix[i][parent[i]]);

    sort(edge_list.begin(), edge_list.end(),
         [](const Edge& a, const Edge& b) { return get<2>(a) > get<2>(b); });

    vector<pair<int, int>> edge_nodes;
    vector<double> edge_weights;
    for (const auto& [u, v, w] : edge_list) {
        edge_nodes.emplace_back(u, v);
        edge_weights.push_back(w);
    }

 
    vector<thread> threads;
    for (int t = 0; t < n_jobs; ++t)
        threads.emplace_back(worker, t,
                                cref(base_mst),
                                cref(edge_nodes),
                                cref(edge_weights),
                                ref(mmj_matrix),
                                ref(task_queue),
                                ref(queue_mutex));
 
 
    auto start3 = chrono::high_resolution_clock::now();

    for (auto& th : threads) th.join();

    auto end3 = chrono::high_resolution_clock::now();

    // cout << "Time used waiting threads finish: " << chrono::duration<double>(end3 - start3).count() << " seconds\n";
    return mmj_matrix;
}

int main() {
    int N = 10000;
    int seed = 222;
    int n_jobs = thread::hardware_concurrency();
    cout << N << endl;
    cout << n_jobs << endl;

    auto distanceMatrix = createDistanceMatrix(N, seed);

    cout << N << endl;

    auto start = chrono::high_resolution_clock::now();
    auto mmjMatrix = cal_mmj_matrix_by_algo_4_Calculation_and_Copy_parallel_compu(distanceMatrix, n_jobs);
    auto end = chrono::high_resolution_clock::now();

    cout << "Time used (Algorithm 4): " << chrono::duration<double>(end - start).count() << " seconds\n";
    cout << "Print last 30 values of the first row of mmj matrix: " << endl;
    const auto& row = mmjMatrix[0];
    for (size_t i = row.size() - 30; i < row.size(); ++i)
        cout << fixed << setprecision(1) << row[i] << " ";
    cout << "\n";

    return 0;
}