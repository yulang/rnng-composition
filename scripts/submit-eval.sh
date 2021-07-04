#!/bin/sh

sbatch -p contrib-cpu-long -c8 -o ../log/$1 ./run-eval.sh