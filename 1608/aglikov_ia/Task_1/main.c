#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void Initialization(int *vector, int size);
int FindMin(int *vector, int part);

int main(int argc, char* argv[]) {
    int ProcNum, ProcRank;
    int TotalMin, ProcMin, size;
    double timeStart, timeEnd;
    MPI_Status Status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0)
        size = (argc != 1) ? atoi(argv[1]) : 75;  // NoArgumetns means size = 75
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int part = size / ProcNum;

    if (ProcRank == 0) {
        int* vector = malloc(size * sizeof(int));
        Initialization(vector, size);
        timeStart = MPI_Wtime();
        for (int i = 1; i < ProcNum - 1; i++)
            MPI_Send(vector + part * i, part, MPI_INT, i, 0, MPI_COMM_WORLD);
        if (ProcNum > 1)
            MPI_Send(vector + part * (ProcNum - 1), size - (ProcNum - 1) * part,
                MPI_INT, ProcNum - 1, 0, MPI_COMM_WORLD);
        ProcMin = FindMin(vector, part);
        free(vector);
    }
    else {
        if (ProcRank == ProcNum - 1)
            part = size - (ProcNum - 1) * part;
        int *dummyVector = malloc(part * sizeof(int));
        MPI_Recv(dummyVector, part, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
        ProcMin = FindMin(dummyVector, part);
        free(dummyVector);
    }

    MPI_Reduce(&ProcMin, &TotalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Print result
    if (ProcRank == 0) {
        timeEnd = MPI_Wtime();
        printf("\nMax element - %d\nTime spent - %.6f", TotalMin, timeEnd - timeStart);
    }
    MPI_Finalize();
    return 0;
}

void Initialization(int *x, int size) {
    srand(time(NULL));
    if (size <= 75)
        for (int i = 0; i < size; i++) {
            x[i] = rand();
           printf("%6d  ", x[i]);  // print if size <= 75 - DONE
        }
    else
        for (int i = 0; i < size; i++)
            x[i] = rand();
}

int FindMin(int *vector, int part) {
    int min = vector[0];
    for (int i = 1; i < part; i++)
        if (vector[i] < min)
            min = vector[i];
    return min;
}
