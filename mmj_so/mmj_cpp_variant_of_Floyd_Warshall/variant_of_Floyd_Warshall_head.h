#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>       
#include <algorithm>     
#include <list> 

 
using namespace std;
 

void cal_mmj_matrix_smart(double * dis_matrix, double * mmj_matrix, int N){
    // vector<double> n_mmj_to_n_minus_1;

    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) 
            if (i != j)
                mmj_matrix[i*N+j] = dis_matrix[i*N+j];

    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) 
            if (i != j)
                for (int k=0;k<N;k++) 
                   if ((i != k) && (j != k))
                        mmj_matrix[j*N+k] = min (mmj_matrix[j*N+k], max (mmj_matrix[j*N+i], mmj_matrix[i*N+k])); }


  