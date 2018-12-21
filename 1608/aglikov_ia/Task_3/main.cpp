// 31. Linear image filtering (vertical split). The Gaussian kernel 3x3.

#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "mpi.h"

void printImage(int* image, int width, int height);
void createKernel(double* kernel, int radius, int size, double sigma);
void createCheckImage(int* originIm, int* checkImage, double* kernel, int height, int width, int size, int radius);
int Clamp(int value, int min, int max);
void equality_check(int * res, int * res2, int height, int width);

int main(int argc, char* argv[]) {
    srand(time(NULL));
    int ProcNum, ProcRank;
    int width, height, kernelRadius, size;
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
        double sigma = (argc >= 4) ? atof(argv[3]) : 1.0;
        kernelRadius = ((argc >= 5) && (argv[4] > 0)) ? atoi(argv[4]) : 1;

        size = 2 * kernelRadius + 1;
        image = new int[width * height];
        kernel = new double[size * size];
        checkImage = new int[width * height];

        // Image initialization
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                image[i * width + j] = rand() % 100 + 1;
        std::cout << "Original image:\n";
        printImage(image, width, height);  // Print image

        createKernel(kernel, kernelRadius, size, sigma);  // Kernel initialization

        // Image for checking
        createCheckImage(image, checkImage, kernel, height, width, size, kernelRadius);
        std::cout << "Image for checking:\n";
        printImage(checkImage, width, height); // Print image for checking

        endTime1 = MPI_Wtime();
    }
    
    // Parallel algorithm
    if (ProcRank == 0)
        startTime2 = MPI_Wtime();
    
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernelRadius, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ProcRank != 0) {
        image = new int[width * height];
        kernel = new double[size * size];
    }
    MPI_Bcast(kernel, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Datatype type1, type2;

    int count = (width / ProcNum + 2);
    MPI_Type_contiguous(count, MPI_INT, &type2);
    MPI_Type_commit(&type2);

    int stride = sizeof(int) * width;

    MPI_Type_create_hvector(height, 1, stride, type2, &type1);
    MPI_Type_commit(&type1);

    int tmp = width - ((width / ProcNum) * (ProcNum - 2) + (width / ProcNum + 1)) - 1;

    if (ProcRank == 0)
        for (int i = 1; i < ProcNum; i++)
            MPI_Send((image + ((tmp - i + 1) + (i - 1) * (width / ProcNum + 1))), 1, type1, i, NULL, MPI_COMM_WORLD);
    else
        MPI_Recv(image, 1, type1, 0, NULL, MPI_COMM_WORLD, &status);
    int temp = count;
    if (ProcRank == 0)
        temp = tmp + 2;

    int* result = new int[height * temp];

    double qwerty;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < temp; j++) {
            qwerty = 0;
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++) {
                    int idY = Clamp(i + y, 0, height - 1);
                    int idX = Clamp(j + x, 0, temp - 1);
                    qwerty += image[idY * height + idX] * kernel[(y + kernelRadius) * size + x + kernelRadius];
                }
            result[i * temp + j] = Clamp(round(qwerty), 0, 255);
        }


    if (ProcRank != 0)
        MPI_Send(result, height*count, MPI_INT, 0, NULL, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        temp--;
        int* res = new int[height * count];
        for (int s = 1; s < ProcNum; s++) {
            MPI_Recv(res, height * count, MPI_INT, s, NULL, MPI_COMM_WORLD, &status);

            for (int i = 0; i < height; i++)
                for (int j = 0; j < count; j++)
                    image[i * height + (j + temp + (count - 2) * ((s - 1)))] = res[i*count + j + 1];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < temp; j++)
                    image[i * height + j] = result[i * temp + j];
        }
        int tempbus = temp;
        temp++;
        if (ProcNum == 1)
            tempbus++;

        for (int i = 0; i < height; i++)
            for (int j = 0; j < tempbus; j++)
                image[i * height + j] = result[i * temp + j];

        endTime2 = MPI_Wtime();
    }
    if (ProcRank == 0) {
        std::cout << "Image made with parallel algorithm:\n";
        printImage(image, width, height);
        equality_check(image, checkImage, height, width);
    }

    if (ProcRank == 0) {
        std::cout << std::fixed << "Non-parallel algorithm time: "
            << (endTime1 - startTime1) << "\nParallel algorithm time: "
            << (endTime2 - startTime2) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void printImage(int* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            std::cout << image[i * width + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void createKernel(double* kernel, int radius, int size, double sigma) {
    double norm = 0;
    for (int i = -radius; i <= radius; i++)
        for (int j = -radius; j <= radius; j++) {
            kernel[(i + radius) * size + (j + radius)] = exp(-(i * i + j * j) / (sigma * sigma));
            norm += kernel[(i + radius) * size + (j + radius)];
        }
    for (int i = 0; i < size; i++)  // Normalization
        for (int j = 0; j < size; j++)
            kernel[i * size + j] /= norm;
    // Print kernel
    std::cout << "Kernel:\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            std::cout << std::fixed << kernel[i * size + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
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

void equality_check(int * res, int * res2, int height, int width) {
    bool flag = false;
    for (int i = 0; i < height*width; i++)
        if (res2[i] - res[i] != 0) {
            flag = !flag;
            break;
        }
    if (!flag)
        std::cout << "The program works correctly.\n";
}
