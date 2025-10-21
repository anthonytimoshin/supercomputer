# supercomputers
Laboratory works on the course supercomputers (parallel programming) in the 5th semester of the MAI.  
Lecturer: Yuri Titov. Accepted by Dmitry Parfenyuk.

run knn_algorithm.f90:
```
gfortran knn_algorithm.f90 -o knn_algorithm
./knn_algorithm
```

run knn_openmp.f90:
```
gfortran -fopenmp knn_openmp.f90 -o knn_openmp
./knn_openmp
```

run knn_cuda.cuf:

on Apple MacBook Pro M1 there are no available NVIDIA GPUs to run CUDA Fortran files.  
So I have another solution. I rent virtual server [here](https://intelion.cloud) with:  
* NVIDIA Tesla A10 24GB;  
* 18 vCPU;  
* 32 GB DDR4 ECC;  
* 256 GB NVMe Gen4;  
* Ubuntu 24.04 + CUDA 12.8.  

```
nvfortran -acc -gpu=cc80 -cuda knn_cuda.cuf -o knn_cuda
./knn_cuda
```

run knn_mpi.f90:
```
mpif90 -o knn_mpi knn_mpi.f90
mpirun -np 4 ./knn_mpi
```
