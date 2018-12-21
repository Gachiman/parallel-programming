// 31. Linear image filtering (vertical split). The Gaussian kernel 3x3.

#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "mpi.h"

void createCheckImage(int* originIm, int* checkImage, double* kernel, int height, int width, int size, int radius);
int Clamp(int value, int min, int max);

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int ProcNum, ProcRank;
    int width, height, kernelRadius;
    double sigma;
    double* kernel = nullptr;
    int* image = nullptr;
    int* checkImage = nullptr;  // For checking
    double startTime1, startTime2, endTime1, endTime2;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    // Non-parallel algorithm

    if (ProcRank == 0) {
        startTime1 = MPI_Wtime();

        // Input parametrs
        width = ((argc >= 2) && (atoi(argv[1]) > 0)) ? atoi(argv[1]) : 3;
        height = ((argc >= 3) && (atoi(argv[2]) > 0)) ? atoi(argv[2]) : 3;
        sigma = (argc >= 4) ? atof(argv[3]) : 1.0;
        kernelRadius = ((argc >= 5) && (argv[4] > 0)) ? atoi(argv[4]) : 1;

        image = new int[width * height];
        checkImage = new int[width * height];

        // Image initialization
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                //image[i * width + j] = j;
                image[i * width + j] = rand() % 100 + 1;
        // Print image
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
                std::cout << image[i * width + j] << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Kernel initialization
        int size = 2 * kernelRadius + 1;
        kernel = new double[size * size];
        double norm = 0;
        for (int i = -kernelRadius; i <= kernelRadius; i++)
            for (int j = -kernelRadius; j <= kernelRadius; j++) {
                kernel[(i + kernelRadius) * size + (j + kernelRadius)] = exp(-(i * i + j * j) / (sigma * sigma));
                norm += kernel[(i + kernelRadius) * size + (j + kernelRadius)];
            }
        for (int i = 0; i < size; i++)  // Normalization
            for (int j = 0; j < size; j++)
                kernel[i * size + j] /= norm;
        // Print kernel
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                std::cout << std::fixed << kernel[i * size + j] << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;

        createCheckImage(image, checkImage, kernel, height, width, size, kernelRadius);

        // Print image for checking
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
                std::cout << checkImage[i * width + j] << ' ';
            std::cout << std::endl;
        }
        std::cout << std::endl;

        endTime1 = MPI_Wtime();
    }

    // Parallel algorithm

    if (ProcRank == 0) {
        std::cout << std::fixed << (endTime1 - startTime1) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void createCheckImage(int* originIm, int* checkImage, double* kernel, int height, int width, int size, int radius) {
    double tmp;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            tmp = 0;
            for (int y = -radius; y <= radius; y++)
                for (int x = -radius; x <= radius; x++) {
                    int idY = Clamp(i + y, 0, height - 1);
                    int idX = Clamp(j + x, 0, width - 1);
                    tmp += originIm[idY * height + idX] * kernel[(y + radius) * size + x + radius];
                }
            checkImage[i * width + j] = Clamp(round(tmp), 0, 255);
        }
}

int Clamp(int value, int min, int max) {
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}
