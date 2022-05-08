#!/bin/bash
#PBS -N add_cuda_job ### Job name 
PBS -l walltime=00:00:30 ### Expected job maximum duration 
#PBS -l nodes=1:ppn=1:gpus=1 ### Computing resources needed 
#PBS -q fast ### Queue 
#PBS -j oe ### Combine standard output and error in the same file

module load cuda/cuda-8.0

#nvcc -g -G add.cu -o add.out

# run program
./add.out > output.log
