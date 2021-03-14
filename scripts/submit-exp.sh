#!/bin/sh

sbatch -p contrib-cpu -c8 -o ../log/$1 ./run-exp.sh