#!/bin/sh
### Name
#BSUB -J create_dataset
#BSUB -q hpc
### Num cores
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
### Memory per core
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo coral_dataset.out
#BSUB -eo coral_dataset.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate
python create_mp3_files_coral.py