#include "../include/OmpHello.h"

using std::cout;
using std::endl;

void OmpHello::hello()
{
    int nThreads = omp_get_max_threads();
    omp_set_num_threads(nThreads);
    cout << "OpenMP with " << nThreads << " threads." << endl;
    
    #pragma omp parallel // private(tID) 
    {
        cout << "Hello from thread " << omp_get_thread_num() << endl; 
    }



}
