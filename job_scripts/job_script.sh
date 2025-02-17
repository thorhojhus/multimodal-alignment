#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J mm_train_2b
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
#BSUB -o /work3/s224233/shared/multimodal-alignment/job_outs/gpu-%J.out
#BSUB -oo latest_gemma.out
#BSUB -e /work3/s224233/shared/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo latest_gemma.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate
#python train.py model.lm_model_name=google/gemma-2-9b-it model.encoder_model_name=syvai/hviske-v2 training.batch_size=32 dataset.name=coral model.load_in_4bit=true wandb.enabled=true training.resume_from_checkpoint=true training.resume_checkpoint_path=/work3/s224233/shared/multimodal-alignment/outputs/2025-02-14/13-36-39/checkpoints/checkpoint_latest.pt

python train.py model.lm_model_name=google/gemma-2-2b-it model.encoder_model_name=openai/whisper-base training.batch_size=64 dataset.name=coral model.load_in_4bit=true wandb.enabled=true