// 31. Linear image filtering (vertical split). The Gaussian kernel 3x3.

#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <mpi.h>

void printImage(unsigned char* image, int width, int height);
void createKernel(float* kernel, int radius, int size, float sigma);
void processImage(unsigned char* originIm, unsigned char* checkImage,
    float* kernel, int height, int width, int size, int radius);
unsigned char Clamp(float value);
void equality_check(unsigned char* res, unsigned char* res2, int height, int width);

int main(int argc, char** argv) {
    srand(time(NULL));
    int ProcNum, ProcRank;
    int width, height, kernelRadius, size;
    float* kernel = nullptr;
    unsigned char* image = nullptr;
    unsigned char* checkImage = nullptr;  // For checking
    unsigned char* partImage = nullptr;
    unsigned char* res = nullptr;
    unsigned char* result = nullptr;
    double sequentialTime, parallelTime;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    ///////////////////////////////////////////////////////////////////////////
    // Non-parallel algorithm
    ///////////////////////////////////////////////////////////////////////////

    if (ProcRank == 0) {
        // Input parametrs
        width = ((argc >= 2) && (atoi(argv[1]) > 0)) ? atoi(argv[1]) : 10;
        height = ((argc >= 3) && (atoi(argv[2]) > 0)) ? atoi(argv[2]) : 10;
        float sigma = (argc >= 4) ? atof(argv[3]) : 1.0;
        kernelRadius = ((argc >= 5) && (argv[4] > 0)) ? atoi(argv[4]) : 1;

        size = 2 * kernelRadius + 1;
        image = new unsigned char[width * height];
        kernel = new float[size * size];
        checkImage = new unsigned char[width * height];

        // Image initialization
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                image[i * width + j] = rand() % 255;
        if (height * width <= 100) {
            std::cout << "Original image:\n";
            printImage(image, width, height);  // Print image
        }

        createKernel(kernel, kernelRadius, size, sigma);  // Kernel initialization

        // Sequential process
        sequentialTime = MPI_Wtime();
        processImage(image, checkImage, kernel, height, width, size, kernelRadius);
        sequentialTime = MPI_Wtime() - sequentialTime;

        // Image for checking
        if (height * width <= 100) {
            std::cout << "Image for checking:\n";
            printImage(checkImage, width, height); // Print image for checking
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Parallel algorithm
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // Preparing

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernelRadius, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ProcRank != 0)
        kernel = new float[size * size];
    MPI_Bcast(kernel, size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);


    int* part = new int[ProcNum];
    int* partStep = new int[ProcNum];
    int* step = new int[ProcNum];
    if (ProcRank == 0) {
        for (int i = 0; i < ProcNum; i++)  // step
            step[i] = width / ProcNum;
        int x = width % ProcNum;
        while (x > 0)
            step[x--]++;
        partStep[0] = 0;  // partStep
        for (int i = 1; i < ProcNum; i++)
            partStep[i] = partStep[i - 1] + step[i - 1];
        for (int i = 0; i < ProcNum; i++)  // part
            part[i] = step[i] * height;
    }
    MPI_Bcast(step, ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(part, ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(partStep, ProcNum, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Datatype mpi_LANE1;
    MPI_Type_vector(height, step[0], width, MPI_UNSIGNED_CHAR, &mpi_LANE1);
    MPI_Type_commit(&mpi_LANE1);

    MPI_Datatype mpi_LANE2;
    MPI_Type_vector(height, step[0] + 1, width, MPI_UNSIGNED_CHAR, &mpi_LANE2);
    MPI_Type_commit(&mpi_LANE2);

    MPI_Datatype mpi_LANE3;
    MPI_Type_vector(height, step[0] + 2, width, MPI_UNSIGNED_CHAR, &mpi_LANE3);
    MPI_Type_commit(&mpi_LANE3);

    MPI_Datatype mpi_LANE4;
    MPI_Type_vector(height, step[0] + 3, width, MPI_UNSIGNED_CHAR, &mpi_LANE4);
    MPI_Type_commit(&mpi_LANE4);

    MPI_Datatype mpi_LANE5;
    MPI_Type_vector(height, step[0], step[0] + 1, MPI_UNSIGNED_CHAR, &mpi_LANE5);
    MPI_Type_commit(&mpi_LANE5);

    MPI_Datatype mpi_LANE6;
    MPI_Type_vector(height, step[0], step[0] + 2, MPI_UNSIGNED_CHAR, &mpi_LANE6);
    MPI_Type_commit(&mpi_LANE6);

    MPI_Datatype mpi_LANE7;
    MPI_Type_vector(height, step[0] + 1, step[0] + 2, MPI_UNSIGNED_CHAR, &mpi_LANE7);
    MPI_Type_commit(&mpi_LANE7);

    MPI_Datatype mpi_LANE8;
    MPI_Type_vector(height, step[0] + 1, step[0] + 3, MPI_UNSIGNED_CHAR, &mpi_LANE8);
    MPI_Type_commit(&mpi_LANE8);

    result = new unsigned char[width * height];
    if (ProcRank == 0 || ProcRank == (ProcNum - 1))
        partImage = new unsigned char[part[ProcRank] + height];
    else
        partImage = new unsigned char[part[ProcRank] + 2 * height];

    /////////////////////////////////////////////////////////////////////////////

    MPI_Barrier(MPI_COMM_WORLD);

    if (ProcRank == 0) {
        parallelTime = MPI_Wtime();

        MPI_Sendrecv(image, 1, mpi_LANE2, 0, 0,
            partImage, part[0] + height, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        for (int i = 1; i < ProcNum - 1; i++)
            if (step[i] == step[0])
                MPI_Send(image + (partStep[i] - 1), 1, mpi_LANE3, i, 0, MPI_COMM_WORLD);
            else
                MPI_Send(image + (partStep[i] - 1), 1, mpi_LANE4, i, 0, MPI_COMM_WORLD);
        if (step[ProcNum - 1] == step[0])
            MPI_Send(image + (partStep[ProcNum - 1] - 1), 1, mpi_LANE2, ProcNum - 1, 0, MPI_COMM_WORLD);
        else
            MPI_Send(image + (partStep[ProcNum - 1] - 1), 1, mpi_LANE3, ProcNum - 1, 0, MPI_COMM_WORLD);
    }
    else {
        if (ProcRank == ProcNum - 1)
            MPI_Recv(partImage, part[ProcRank] + height, MPI_UNSIGNED_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        else
            MPI_Recv(partImage, part[ProcRank] + 2 * height , MPI_UNSIGNED_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }

    /////////////////////////////////////////////////////////////////////////////

    if (ProcRank == 0 || ProcRank == (ProcNum - 1))
        res = new unsigned char[part[ProcRank] + height];
    else
        res = new unsigned char[part[ProcRank] + 2 * height];

    if (ProcRank == 0 || ProcRank == ProcNum - 1)
        processImage(partImage, res, kernel, height, step[ProcRank] + 1, size, kernelRadius);
    else
        processImage(partImage, res, kernel, height, step[ProcRank] + 2, size, kernelRadius);

    /////////////////////////////////////////////////////////////////////////////

    if (ProcRank == 0) {
        MPI_Sendrecv(res, 1, mpi_LANE5, 0, 0,
            result, 1, mpi_LANE1, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        for (int i = 1; i < ProcNum - 1; i++)
            if (step[i] == step[0])
                MPI_Recv(result + partStep[i], 1, mpi_LANE1, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            else
                MPI_Recv(result + partStep[i], 1, mpi_LANE2, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        if (step[ProcNum - 1] == step[0])
            MPI_Recv(result + partStep[ProcNum - 1], 1, mpi_LANE1, ProcNum - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        else
            MPI_Recv(result + partStep[ProcNum - 1], 1, mpi_LANE2, ProcNum - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }
    else {
        if (ProcRank == ProcNum - 1)
            if (step[ProcNum - 1] == step[0])
                MPI_Send(res + 1, 1, mpi_LANE5, 0, 0, MPI_COMM_WORLD);
            else
                MPI_Send(res + 1, 1, mpi_LANE7, 0, 0, MPI_COMM_WORLD);
        else
            if (step[ProcRank] == step[0])
                MPI_Send(res + 1, 1, mpi_LANE6, 0, 0, MPI_COMM_WORLD);
            else
                MPI_Send(res + 1, 1, mpi_LANE8, 0, 0, MPI_COMM_WORLD);
    }

    /////////////////////////////////////////////////////////////////////////////

    if (ProcRank == 0) {  // Print conclusion
        parallelTime = MPI_Wtime() - parallelTime;
        if (height * width <= 100) {
            std::cout << "Image made with parallel algorithm:\n";
            printImage(result, width, height);
        }
        equality_check(result, checkImage, height, width);  // Test for equality
        std::cout << std::fixed << "Non-parallel algorithm time: "
            << sequentialTime << "\nParallel algorithm time: "
            << parallelTime << std::endl;
    }
    MPI_Finalize();
    return 0;
}

void printImage(unsigned char* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            std::cout << image[i * width + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void createKernel(float* kernel, int radius, int size, float sigma) {
    float norm = 0;
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

void processImage(unsigned char* originIm, unsigned char* checkImage, float* kernel, int height, int width, int size, int radius) {
    float tmp;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            tmp = 0;
            for (int y = -radius; y <= radius; y++)
                for (int x = -radius; x <= radius; x++)
                    if ((i + y) >= 0 && (i + y) < height && (j + x) >= 0 && (j + x) < width)
                        tmp += originIm[(i + y) * width + j + x] * kernel[(y + radius) * size + x + radius];
            checkImage[i * width + j] = Clamp(roundf(tmp));
        }
}

unsigned char Clamp(float value) {
    if (value < 0)    return 0;
    if (value > 255)    return 255;
    return (unsigned char)value;
}

void equality_check(unsigned char * res, unsigned char * res2, int height, int width) {
    bool flag = false;
    for (int i = 0; i < height * width; i++)
        if (res2[i] != res[i]) {
            flag = !flag;
            break;
        }
    if (!flag)
        std::cout << "The program works correctly.\n";
}
