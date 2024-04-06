#include "../include/PrintVector.h"

using std::cout;
using std::endl;
using std::complex;
using std::vector;

template <typename T>
void PrintVector<T>::printVector(const std::vector<T>& v) const
{
    cout << "**************************************" << endl;
    //for (const auto& i : v)
    //{
     //   cout << i << " ";
    //}
    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            cout << v[i] << " ";
        }
        cout << endl;
    }
    cout << "\n**************************************" << endl;
}

template<>
void PrintVector<complex<float>>::printVector(const std::vector<complex<float>>& v) const
{
    cout << "**************************************" << endl;
    for (const auto& i : v)
    {
        cout << "(" << i.real() << " + " << i.imag() << "i) ";
    }
    cout << "\n**************************************" << endl;
}

template<>
void PrintVector<complex<double>>::printVector(const std::vector<complex<double>>& v) const
{
    cout << "**************************************" << endl;
    for (const auto& i : v)
    {
        cout << "(" << i.real() << " + " << i.imag() << "i) ";
    }
    cout << "\n**************************************" << endl;
}

template void PrintVector<float>::printVector(const std::vector<float>&) const;
template void PrintVector<double>::printVector(const std::vector<double>&) const;
template void PrintVector<int>::printVector(const std::vector<int>&) const;