#!/bin/sh

sbatch -p contrib-cpu -c4 -o ../log/$1 ./run-exp.sh