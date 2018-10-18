#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "mpi.h"

#define N 300

void DataInitialization(int *x);

int main(int argc, char* argv[]) {
    int vector[N], ProcMin = INT_MAX, TotalMin;
    int ProcRank, ProcNum;
    double timeStart, timeEnd;
    MPI_Status Status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    timeStart = MPI_Wtime();
    if (ProcRank == 0)
        DataInitialization(vector);
    MPI_Bcast(&vector, N, MPI_INT, 0, MPI_COMM_WORLD);

    int part = N / ProcNum;
    int i1 = part * ProcRank;
    int i2 = (ProcRank == ProcNum - 1)? N : part * (ProcRank + 1);
    for (; i1 < i2; i1++)
        if (ProcMin > vector[i1])
            ProcMin = vector[i1];

    if (ProcRank == 0)       // Process 0
        TotalMin = ProcMin;
    else                     // Others process
        MPI_Send(&ProcMin, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    MPI_Reduce(&ProcMin, &TotalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    timeEnd = MPI_Wtime();

    // Print result
    if (ProcRank == 0)
        printf("\nTotal minimum = %d, work time = %f\n", TotalMin, timeEnd - timeStart);
    MPI_Finalize();
    return 0;
}

void DataInitialization(int *x) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        x[i] = rand();
        printf("%6d  ", x[i]);
    }
}