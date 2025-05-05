#!/bin/sh
### GPU
#BSUB -q gpua100
### Name
#BSUB -J hviske_bench_nst
### Num cores
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
### Memory per core
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo hviske_bench_nst.out
#BSUB -eo hviske_bench_nst.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd ~/multimodal-alignment
source .venv/bin/activate
python hviske_nst.py