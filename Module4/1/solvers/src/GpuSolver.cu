#include "../include/GpuSolver.h"

template<typename T>
inline void GpuSolver<T>::deviceAllocations()
{
	gpuMalloc(&dA, SIZE_A);
	gpuMalloc(&dRowSums, SIZE_SUMS);
	gpuMalloc(&dColSums, SIZE_SUMS);
	gpuCheckErrors("gpuMalloc failure");
}

template<typename T>
void GpuSolver<T>::copyH2D()
{
	gpuMemcpy(dA, this->A.data(), SIZE_A, gpuMemcpyHostToDevice);
	gpuMemcpy(dRowsSums, this->RowsSums.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuMemcpy(dColsSums, this->ColsSums.data(), SIZE_SUMS, gpuMemcpyHostToDevice);
	gpuCheckErrors("gpuMemcpy H2D failure");
}

template<typename T>
void GpuSolver<T>::copyD2H()
{
	gpuMemcpy(this->RowsSums.data(), dRowsSums, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuMemcpy(this->ColsSums.data(), dColsSums, SIZE_SUMS, gpuMemcpyDeviceToHost);
	gpuCheckErrors("gpuMemcpy D2H failure");
}

template<typename T>
GpuSolver<T>::~GpuSolver()
{
	gpuFree(dA);
	gpuFree(dRowsSums);
	gpuFree(dColsSums);
	gpuCheckErrors("gpuFree failure");
}

template<typename T>
void GpuSolver<T>::rowSums()
{
	gpuReportDevice();
	deviceAllocations();
	copyH2D();

}

template<typename T>
void GpuSolver<T>::colSums()
{

}

template void GpuSolver<float>::deviceAllocations();
template void GpuSolver<double>::deviceAllocations();
template void GpuSolver<float>::copyH2D();
template void GpuSolver<double>::copyH2D();
template void GpuSolver<float>::copyD2H();
template void GpuSolver<double>::copyD2H();
template void GpuSolver<float>::rowSums();
template void GpuSolver<double>::rowSums();
template void GpuSolver<float>::colSums();
template void GpuSolver<double>::colSums();
template GpuSolver<float>::~GpuSolver();
template GpuSolver<double>::~GpuSolver();







