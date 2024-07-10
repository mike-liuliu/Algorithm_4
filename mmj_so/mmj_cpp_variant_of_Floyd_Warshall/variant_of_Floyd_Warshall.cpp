#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <iomanip>
#include <iterator>
#include <list>
#include <ctime>

 
#include "variant_of_Floyd_Warshall_head.h"
 
using namespace std;

extern "C" {   
void cal_mmj_matrix(double *  dis_matrix, double * mmj_matrix, int N)
{
//     int kk = N;
cal_mmj_matrix_smart(dis_matrix, mmj_matrix,N);
}
}
