#include "../include/GpuHello.h"

__global__
void helloWorld()
{
	printf("Hello from block: %d, thread: %d\n", blockIdx.x, threadIdx.x);
}

void GpuHello::hello()
{
	gpuReportDevice();
	helloWorld << < GRID_SIZE, BLOCK_SIZE >> > ();
	gpuDeviceSynchronize();
}
