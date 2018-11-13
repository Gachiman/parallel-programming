// №5 Передача от одного всем (broadcast)

#include <iostream>
#include <cmath>
#include "mpi.h"

void MPI_Broadcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);  // Tree

void MPI_Broadcast0(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);  // Not tree

int bit(int x);  // Finding the high bit position

int main(int argc, char* argv[]) {
    int ProcRank, ProcNum;
    int size, procRoot;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (ProcRank == 0) {
        procRoot = (argc > 1) ? atoi(argv[1]) : 0;
        if (procRoot < 0) procRoot = 0;
        else if (procRoot >= ProcNum) procRoot = ProcNum - 1;
    }

    MPI_Broadcast(&procRoot, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (ProcRank == procRoot) {
        size = (argc == 3) ? atoi(argv[2]) : 10;
        std::cout << "Process " << procRoot << " broadcasting data " 
            << size << std::endl;
        MPI_Broadcast(&size, 1, MPI_INT, procRoot, MPI_COMM_WORLD);
    }
    else {
        MPI_Broadcast(&size, 1, MPI_INT, procRoot, MPI_COMM_WORLD);
        std::cout << "Process " << ProcRank << " received data " << size
            << " from process " << procRoot << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void MPI_Broadcast(void *buf, int count, MPI_Datatype type, int root,
                   MPI_Comm comm) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(comm, &ProcRank);
    MPI_Comm_size(comm, &ProcNum);
    int procGen = bit(ProcRank);

    if (ProcRank != root)
        MPI_Recv(buf, count, type, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

    for (int i = procGen; i < bit(ProcNum - 1); i++) {
        if (ProcRank + pow(2, i) >= ProcNum)
            break;
        MPI_Send(buf, count, type, ProcRank + pow(2, i), 0, comm);
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

int bit(int x) {
    int pos = 0;
    while (x != 0) {
        x >>= 1;
        pos++;
    }
    return pos;
}
