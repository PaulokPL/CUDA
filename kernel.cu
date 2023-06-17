#include <iostream>
#include <fstream>
#include <cstring>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iomanip>
#include <math.h>
#include <cmath>
#include <cassert>
#include <stdlib.h> 

#define BLOCK_SIZE 32
#define SHARED_MEM_SIZE 4000*4


// Structure to store WAV header information
struct WAVHeader {
    char chunkID[4];
    int chunkSize;
    char format[4];
    char subchunk1ID[4];
    int subchunk1Size;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    char subchunk2ID[4];
    int subchunk2Size;
};

void writeWAVHeader(std::ofstream& outputFile, uint32_t sampleRate, uint16_t numChannels, uint16_t bitsPerSample, uint32_t dataSize) {
    WAVHeader header;

    header.chunkID[0] = 'R';
    header.chunkID[1] = 'I';
    header.chunkID[2] = 'F';
    header.chunkID[3] = 'F';
    header.chunkSize = 36 + dataSize;
    header.format[0] = 'W';
    header.format[1] = 'A';
    header.format[2] = 'V';
    header.format[3] = 'E';
    header.subchunk1ID[0] = 'f';
    header.subchunk1ID[1] = 'm';
    header.subchunk1ID[2] = 't';
    header.subchunk1ID[3] = ' ';
    header.subchunk1Size = 16;
    header.audioFormat = 1;
    header.numChannels = numChannels;
    header.sampleRate = sampleRate;
    header.byteRate = sampleRate * numChannels * bitsPerSample / 8;
    header.blockAlign = numChannels * bitsPerSample / 8;
    header.bitsPerSample = bitsPerSample;
    header.subchunk2ID[0] = 'd';
    header.subchunk2ID[1] = 'a';
    header.subchunk2ID[2] = 't';
    header.subchunk2ID[3] = 'a';
    header.subchunk2Size = dataSize;

    outputFile.write((char*)&header, sizeof(header));
}

__global__ void matrixMultiply(float* a, float* b, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * num + col] = 0;
    if ((row < num) && (col < num)) {
        for (int m = 0; m < num; m++)
            c[row * num + col] += a[row * num + m] * b[m * num + col];
    }
}

__global__ void matrixMultiplycd(float* a, float* b, double* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * num + col] = 0;
    if ((row < num) && (col < num)) {
        for (int m = 0; m < num; m++)
            c[row * num + col] += a[row * num + m] * b[m * num + col];
    }
}

__global__ void matrixMultiplyacd(double* a, float* b, double* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * num + col] = 0;
    if ((row < num) && (col < num)) {
        for (int m = 0; m < num; m++)
            c[row * num + col] += a[row * num + m] * b[m * num + col];
    }
}

__global__ void create_matrix_kernel(float* matrix, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
    {
        matrix[idx] = 0.0f;
    }
}

__global__ void przypisanie(float* a, int* b, int w, int z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col == 0 && row == 0) {
        a[z] = b[w];
    }
}

__global__ void przypisanie2(float* a, float* b, int w, int z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col == 0 && row == 0) {
        a[z] = b[w];
    }
}


__global__ void przypisanief(float* a, float* b, int w, int z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col == 0 && row == 0) {
        a[z] = b[w];
    }

}

__global__ void subtract(float* a, float* b, float* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col == 0 && row == 0) {
        c[0] = a[0] - b[0];
    }
}

__global__ void matrixTranspose(float* a, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = a[col * num + row];
}

__global__ void prepar(float* b, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col == 0 && row < num) {
        b[row * num] = 0;
    }
}

__global__ void alfa_mat(float* alfa, int* y_input, int w, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col == 0 && row < num) {
        alfa[row * num] = y_input[w - num + row];
    }
}

__global__ void matrix_P(float* a, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row == col) {
        a[col * 4 + row] = 1000;
    }
    else {
        a[col * 4 + row] = 0;
    }
}

__global__ void addScalarKernel(double* A, float scalar, double* B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 && j == 0)
        B[i * N + j] = A[i * N + j] + scalar;
}


__global__ void standard_deviation_kernel(const float* d_numbers, float* d_mean, float* d_stddev, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    float diff = 0.0f;
    if (i < n)
    {
        sum += d_numbers[i];
    }

    if (threadIdx.x == 0)
    {
        d_mean[0] = sum / n;
    }

    if (i < n)
    {
        diff = d_numbers[i] - d_mean[0];
        sum += diff * diff;
    }

    if (threadIdx.x == 0)
    {
        d_stddev[0] = sqrtf(sum / n);
    }
}

__global__ void matrixSubtract(float* a, float* b, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = a[row * num + col] - b[row * num + col];
}

__global__ void matrixMultiplyByConst(float* a, float* t, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = t[0] * a[row * num + col];
}

__global__ void matrixMultiplyByConstreal(float* a, float t, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = t * a[row * num + col];
}

__global__ void matrixMultiplyByConstd(double* a, float t, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = t * a[row * num + col];
}

__global__ void matrixDivisionKernel(float* A, double* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
        C[i * N + j] = A[i * N + j] / B[0];
}

__global__ void matrixDivisionKerneld(double* A, double* B, double* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
        C[i * N + j] = A[i * N + j] / B[0];
}


__global__ void matrixAdd(float* a, float* b, float* c, int num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < num) && (col < num))
        c[row * num + col] = a[row * num + col] + b[row * num + col];
}


__global__ void checking_if(float* last_err, float* std, int w, float* y_o, int* y, int r, int max) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col == 0) {
        if ((abs(last_err[0]) > std[0]) && (w < max - r)) {
            y_o[0] = (y[w - r] + y[w + r]) / 2;
        }
    }
}

__global__ void circshiftKernel(float* err, int shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < shift) {
        //if (i == 0) {
        //    shifted_err[shift - 1] = err[i];
        //}
        //else {
        //    shifted_err[i] = err[i + 1];
        //}
        int temp = err[i];
        err[i] = err[(i + 1) % shift];
        err[(i + 1) % shift] = temp;
    }
    //int index = (i + shift + 50) % 50;
    //shifted_err[index] = err[i];
}



int main() {
    // Open the WAV file
    std::ifstream file("01.wav", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    // Read the WAV header
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Check if it's a valid WAV file
    if (strncmp(header.chunkID, "RIFF", 4) || strncmp(header.format, "WAVE", 4)) {
        std::cerr << "Error: Not a valid WAV file" << std::endl;
        return 1;
    }

    // Allocate memory for the sample data
    short* sampleData = new short[header.subchunk2Size / 2];
    file.read(reinterpret_cast<char*>(sampleData), header.subchunk2Size);

    // Close the file
    file.close();
    const int len = header.subchunk2Size / 2;
    const int r = 4;
    const int M = 50;
    int* y_inp = new int[header.subchunk2Size / 2 * 1];


    for (int i = 0; i < header.subchunk2Size / 2; i++) {
        y_inp[i] = sampleData[i];
    }

    const size_t bytes = sizeof(float) * r * r;
    const size_t bytes1 = sizeof(float) * header.subchunk2Size / 2;

    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU);
    //Device vector pointers
    float* alfa = new float[r * r];
    float* alfa_transpose = new float[r * r];
    float* beta = new float[r * r];
    float* qwer = new float[r * r];
    float* k = new float[r * r];
    float* k_den = new float[r * r];
    double* k_den1 = new double[1];
    double* qwer1 = new double[r * r];
    float* y_returntr = new float[header.subchunk2Size / 2 * 1];
    float* y_return = new float[header.subchunk2Size / 2 * 1];
    int* y_input = new int[header.subchunk2Size / 2 * 1];
    float* P = new float[r * r];
    float* meter = new float[r * r];
    double* meter1 = new double[r * r];
    float* y_o = new float[1];
    float* y_o1 = new float[1];
    float* mean = new float[1];
    float* stddev = new float[1];
    float* err = new float[1 * M];
    float* y_pred = new float[1];
    float* y_pred1 = new float[1];
    float* last_err = new float[1];
    float* eps = new float[1];
    float* lambda = new float[1];
    lambda[0] = 0.99;
    for (int q = 0; q < r; q++) {
        y_return[q] = y_inp[q];
    }


    cudaMalloc(&alfa, bytes);
    cudaMalloc(&k, bytes);
    cudaMalloc(&k_den, sizeof(float) * r * r);
    cudaMalloc(&mean, sizeof(float));
    cudaMalloc(&stddev, sizeof(float));
    cudaMalloc(&alfa_transpose, bytes);
    cudaMalloc(&beta, bytes);
    cudaMalloc(&y_return, bytes1);
    cudaMalloc(&P, bytes);
    cudaMalloc(&meter, bytes);
    cudaMalloc(&y_input, bytes1);
    cudaMalloc(&err, sizeof(float) * M);
    cudaMalloc(&y_pred, sizeof(float));
    cudaMalloc(&y_o, sizeof(float));
    cudaMalloc(&y_o1, sizeof(float));
    cudaMalloc(&k_den1, sizeof(double));
    cudaMalloc(&meter1, sizeof(double) * r * r);
    cudaMalloc(&last_err, sizeof(float));
    cudaMalloc(&eps, sizeof(float));
    cudaMalloc(&lambda, sizeof(float));
    cudaMemcpy(y_input, y_inp, bytes1, cudaMemcpyHostToDevice);
    //Grid size
    int GRID_SIZE = (int)ceil((float)r / BLOCK_SIZE);
    int GRID_SIZE2 = (int)ceil((float)header.subchunk2Size / 2 / BLOCK_SIZE);

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 grid2(GRID_SIZE2, GRID_SIZE2);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockDim(r, r);
    dim3 gridDim(1, 1);
    dim3 blockDim1(1, 1);
    int block_size = 256;
    int num_blocks = (r * r + block_size - 1) / block_size;
    dim3 blockSize(512);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x);


    prepar << < gridDim, blockDim >> > (beta, r);
    matrix_P << < gridDim, blockDim >> > (P, r);

    for (int w = r; w < header.subchunk2Size / 2; w++) {
        alfa_mat << <gridDim, blockDim >> > (alfa, y_input, w, r);
        przypisanie << <gridDim, blockDim >> > (y_o, y_input, w, 0);
        matrixTranspose << <gridDim, blockDim >> > (alfa, alfa_transpose, r);
        matrixMultiply << <gridDim, blockDim >> > (alfa_transpose, beta, y_pred, r);
        subtract << <gridDim, blockDim >> > (y_o, y_pred, last_err);
        if (w > r) {
            if (w - r < M) {
                standard_deviation_kernel << <gridDim, blockDim >> > (err, mean, stddev, w - r);
            }
            else {
                standard_deviation_kernel << <gridDim, blockDim >> > (err, mean, stddev, M);
            }
            checking_if << <gridDim, blockDim >> > (last_err, stddev, w, y_o, y_input, r, header.subchunk2Size / 2);
        }
        przypisanief << <gridDim, blockDim >> > (y_return, y_o, 0, w);
        subtract << <gridDim, blockDim >> > (y_o, y_pred, eps);
        matrixMultiply << < gridDim, blockDim >> > (P, alfa, k, r);
        matrixMultiply << < gridDim, blockDim >> > (alfa_transpose, P, k_den, r);
        matrixMultiplycd << < gridDim, blockDim >> > (k_den, alfa, k_den1, r);
        addScalarKernel << <gridSize, blockSize >> > (k_den1, 1, k_den1, r);
        matrixDivisionKernel << <gridDim, blockDim >> > (k, k_den1, k, r);
        matrixMultiplyByConst << <gridDim, blockDim >> > (k, eps, k, r);
        matrixAdd << <gridDim, blockDim >> > (beta, k, beta, r);
        matrixMultiply << <gridDim, blockDim >> > (P, alfa, meter, r);
        matrixMultiplycd << <gridDim, blockDim >> > (meter, alfa_transpose, meter1, r);
        matrixDivisionKerneld << <gridDim, blockDim >> > (meter1, k_den1, meter1, r);
        matrixMultiplyByConstd << <gridDim, blockDim >> > (meter1, 1000.0f, meter, r);
        matrixSubtract << <gridDim, blockDim >> > (P, meter, P, r);
        matrixMultiplyByConstreal << <gridDim, blockDim >> > (P, 0.99f, P, r);
        subtract << <gridDim, blockDim >> > (y_o, y_pred, y_o1);
        if (w < M + r) {
            przypisanie2 << <gridDim, blockDim >> > (err, y_o1, 0, w - r);
        }
        else {
            circshiftKernel << <gridSize, blockSize >> > (err, 50);
            przypisanie2 << <gridDim, blockDim >> > (err, y_o1, 0, M - 1);
        }
    }

    cudaMemcpy(y_returntr, y_return, header.subchunk2Size / 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(alfa);
    cudaFree(k);
    cudaFree(k_den);
    cudaFree(mean);
    cudaFree(stddev);
    cudaFree(alfa_transpose);
    cudaFree(beta);
    cudaFree(y_return);
    cudaFree(P);
    cudaFree(meter);
    cudaFree(y_input);
    cudaFree(alfa);
    cudaFree(err);
    cudaFree(y_pred);
    cudaFree(y_o);
    cudaFree(y_o1);
    cudaFree(k_den1);
    cudaFree(meter1);
    cudaFree(last_err);
    cudaFree(eps);
    cudaFree(lambda);

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);
    float czasNaGPU = milliseconds / 1000; //[s]
    for (int q = 0; q < r; q++) {
        y_returntr[q] = y_inp[q];
    }
    printf("GPU execution time: %f  [us]\n", (double)(czasNaGPU));

    int16_t* audioDataInt16 = new int16_t[len];
    for (int i = 0; i < header.subchunk2Size / 2; i++) {
        audioDataInt16[i] = (int16_t)roundf(y_returntr[i]);
    }

    std::ofstream outputFile("output.wav", std::ios::binary);
    writeWAVHeader(outputFile, 22050, 1, 16, header.subchunk2Size / 2 * sizeof(int16_t));
    outputFile.write((char*)audioDataInt16, header.subchunk2Size / 2 * sizeof(int16_t));
    outputFile.close();

    // Deallocate the sample data
    delete[] sampleData;

    return 0;
}