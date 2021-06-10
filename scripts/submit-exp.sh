#!/bin/sh

sbatch -p contrib-cpu-long -c2 -o ../log/$1 ./run-exp.sh