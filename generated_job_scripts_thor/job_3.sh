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
#BSUB -oo generated_job_scripts_thor/runlogs/3.out
#BSUB -e ~/multimodal-alignment/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo generated_job_scripts_thor/runlogs/3.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.6.3

cd ~/multimodal-alignment
source .venv/bin/activate
python train.py model.lm_model_name=Gemma-2-9B model.encoder_model_name=syvai/hviske-v2 model.projector=simple model.downsample_factor=8 dataset.name=coral wandb.enabled=true +wandb.project=benchmarks model.load_in_4_bit=true wandb.run_name="3 - Gemma 2 9B + Hviske V2 + Ultravox Style \(Coral\)"