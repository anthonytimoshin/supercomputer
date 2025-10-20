# supercomputer
Laboratory works on the course Super EVM in the 5th semester of the MAI. Lecturer: Yuri Titov. Accepted by Dmitry Parfenyuk.

run knn_algorithm.f90:
```
gfortran knn_algorithm.f90 -o knn_algorithm
./knn_algorithm
```

run lab_mpi.f90:
```
mpif90 -o lab_mpi lab_mpi.f90
mpirun -np 4 ./lab_mpi
```

run lab_cuda.cuf:

on Apple MacBook Pro M1 there are no available NVIDIA GPUs. 
So I have another solution. I rent virtual server [here](https://intelion.cloud) with:
NVIDIA Tesla A10 24GB;
18 vCPU;
32 GB DDR4 ECC;
256 GB NVMe Gen4;
Ubuntu 24.04 + CUDA 12.8.
