#!/bin/sh
### GPU
#BSUB -q gpua100
### Name
#BSUB -J mm_train
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
#BSUB -o ~/multimodal-alignment/multimodal-alignment/job_outs/gpu-%J.out
#BSUB -oo latest-sweep.out
#BSUB -e ~/multimodal-alignment/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo latest-sweep.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.6.3

cd ~/multimodal-alignment
source .venv/bin/activate
python train_sweep.py model.lm_model_name=google/gemma-2-2b-it model.encoder_model_name=syvai/hviske-v2 training.batch_size=16 model.load_in_4bit=true wandb.enabled=true wandb.run_name=ultravox-downsample-factor-sweep training.num_epochs=1 dataset.name=coral
#python train.py
