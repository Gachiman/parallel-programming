// №5 Передача от одного всем (broadcast)

#include <iostream>
#include <iomanip>
#include "mpi.h"

void MPI_Broadcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);  // Tree
void MPI_BroadcastDemo(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);  // Tree demonstrarion
void MPI_Broadcast0(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);  // Not tree

int bitHi1(int x);  // Finding the high bit position
void Initialization(int* in, float* fl, double* doub, int size);  // Arrays Initialication

int main(int argc, char* argv[]) {
    int ProcRank, ProcNum;
    int size, procRoot;
    double tS1, tS2, tE1, tE2;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (ProcRank == 0) {
        procRoot = (argc > 1) ? atoi(argv[1]) : 0;
        if (procRoot < 0) procRoot = 0;
        else if (procRoot >= ProcNum) procRoot = ProcNum - 1;
        size = (argc == 3) ? atoi(argv[2]) : 5;
    }

    MPI_Broadcast(&procRoot, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Broadcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* vecInt = new int[size];
    float* vecFloat = new float[size];
    double* vecDouble = new double[size];

    if (ProcRank == procRoot) {
        Initialization(vecInt, vecFloat, vecDouble, size);
        tS1 = MPI_Wtime();
    }

    //MPI_Broadcast(vecInt, size, MPI_INT, procRoot, MPI_COMM_WORLD);
    //MPI_Broadcast(vecFloat, size, MPI_FLOAT, procRoot, MPI_COMM_WORLD);
    //MPI_Broadcast(vecDouble, size, MPI_DOUBLE, procRoot, MPI_COMM_WORLD);
    MPI_BroadcastDemo(vecInt, size, MPI_INT, procRoot, MPI_COMM_WORLD);
    if (ProcRank == procRoot)  tE1 = MPI_Wtime();

    if (ProcRank == procRoot) tS2 = MPI_Wtime();  // Time with MPI_Bcast
    MPI_Bcast(vecInt, size, MPI_INT, procRoot, MPI_COMM_WORLD);
    if (ProcRank == procRoot) tE2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (ProcRank == procRoot)
        std::cout << "\nTime spent with castom Bcast = " << std::fixed
            << std::setprecision(5) << tE1 - tS1 << std::endl
            << "Time spent with MPI_Bcast = " << tE2 - tS2 << std::endl;

    MPI_Finalize();
    return 0;
}

void MPI_Broadcast(void *buf, int count, MPI_Datatype type, int root,
                   MPI_Comm comm) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(comm, &ProcRank);
    MPI_Comm_size(comm, &ProcNum);

    if (ProcRank != root)
        MPI_Recv(buf, count, type, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, MPI_STATUSES_IGNORE);
    else if (root != 0)
        MPI_Send(buf, count, type, 0, 0, comm);

    for (int i = bitHi1(ProcRank); ; i++) {
        if ((ProcRank | (1 << i)) >= ProcNum)
            break;
        MPI_Send(buf, count, type, ProcRank | (1 << i), 0, comm);
    }
}

void MPI_BroadcastDemo(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    int ProcRank, ProcNum;
    MPI_Status status;
    MPI_Comm_rank(comm, &ProcRank);
    MPI_Comm_size(comm, &ProcNum);

    if (ProcRank != root) {
        MPI_Recv(buf, count, type, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
        std::cout << "Process " << ProcRank << " received " << count
            << " elements from process " << status.MPI_SOURCE << std::endl;
    }
    else if (root != 0)
        MPI_Send(buf, count, type, 0, 0, comm);

    for (int i = bitHi1(ProcRank); ; i++) {
        if ((ProcRank | (1 << i)) >= ProcNum)
            break;
        MPI_Send(buf, count, type, ProcRank | (1 << i), 0, comm);
    }
}

void MPI_Broadcast0(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(comm, &ProcRank);
    MPI_Comm_size(comm, &ProcNum);

    if (ProcRank == root) {
        for (int i = 0; i < ProcNum; i++)
            if (i != root)
                MPI_Send(buf, count, type, i, 0, comm);
    }
    else
        MPI_Recv(buf, count, type, root, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
}

int bitHi1(int x) {
    int pos = 0;
    while (x != 0) {
        x >>= 1;
        pos++;
    }
    return pos;
}

void Initialization(int *in, float* fl, double* doub, int size) {
    for (int i = 0; i < size; i++) {
        in[i] = i;
        fl[i] = (float)i / 10;
        doub[i] = (double)i / 10;
    }
}
