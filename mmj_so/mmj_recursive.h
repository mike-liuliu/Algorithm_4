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

void cal_mmj_matrix_smart(double * dis_matrix, double * mmj_matrix, int N);
double point_n_mmj_to_each(double * dis_matrix, double * mmj_matrix, int n, int r, int N);
void cal_n_mmj(double * dis_matrix, double * mmj_matrix, int n, vector<double> & n_mmj_to_n_minus_1, int N);
vector<double> point_n_mmj_to_each_of_n_m_1(double * dis_matrix, double * mmj_matrix, int n, int N);
double update_mmj_ij(double * dis_matrix, double * mmj_matrix, int n, vector<double> & n_mmj_to_n_minus_1,int i, int j, int N);



void cal_mmj_matrix_smart(double * dis_matrix, double * mmj_matrix, int N){
    vector<double> n_mmj_to_n_minus_1;

    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) 
            mmj_matrix[i*N+j] = 0;
 

    mmj_matrix[0*N+1] = mmj_matrix[1*N+0] = dis_matrix[0*N+1];
 
    for (int kk=2;kk<N;kk++) {
        // cout << kk << endl;
        n_mmj_to_n_minus_1 = point_n_mmj_to_each_of_n_m_1(dis_matrix, mmj_matrix, kk,N);
        cal_n_mmj(dis_matrix, mmj_matrix, kk, n_mmj_to_n_minus_1,N);        
        }      
        }

 
double point_n_mmj_to_each(double * dis_matrix, double * mmj_matrix, int n, int r, int N){
    vector<double> max_jump_list;
    double m_jump;

    for (int i=0;i<n;i++){
        m_jump = max(dis_matrix[n*N+i],mmj_matrix[i*N+r]);
        max_jump_list.push_back(m_jump);
        }
    return *min_element(max_jump_list.begin(), max_jump_list.end());
    }


void cal_n_mmj(double * dis_matrix, double * mmj_matrix, int n, vector<double> & n_mmj_to_n_minus_1, int N){
    
    for (int i=0;i<n;i++)
        mmj_matrix[i*N+n] = mmj_matrix[n*N+i] = n_mmj_to_n_minus_1[i];
    
    for (int i=0;i<n;i++)        
        for (int j=0;j<n;j++)
            if (i < j)
                mmj_matrix[i*N+j] = mmj_matrix[j*N+i] =  update_mmj_ij(dis_matrix, mmj_matrix, n, n_mmj_to_n_minus_1, i, j,N);                
                } 
 
vector<double> point_n_mmj_to_each_of_n_m_1(double * dis_matrix, double * mmj_matrix, int n, int N){
    vector<double> to_each;
    for (int i=0;i<n;i++)
        to_each.push_back(point_n_mmj_to_each(dis_matrix, mmj_matrix, n, i,N));
    return to_each;}
 


double update_mmj_ij(double * dis_matrix, double * mmj_matrix, int n, vector<double> & n_mmj_to_n_minus_1,int i, int j, int N){
    double m1 = mmj_matrix[i*N+j];
    double m2 = max(n_mmj_to_n_minus_1[i],n_mmj_to_n_minus_1[j]);        
    return min(m1,m2);}

  