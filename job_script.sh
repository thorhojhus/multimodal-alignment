#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J mm_train_9b
### Num cores
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
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
#python train.py
#python train.py model.lm_model_name=HuggingFaceTB/SmolLM2-1.7B-Instruct model.encoder_model_name=openai/whisper-large-v3-turbo training.batch_size=32 wandb.enabled=true
#python train.py model.lm_model_name=google/gemma-2-2b-it model.whisper_model_name=openai/whisper-large-v3-turbo training.batch_size=16 dataset.name=coral
#python train.py model.lm_model_name=google/gemma-2-9b-it model.whisper_model_name=openai/whisper-large-v3-turbo training.batch_size=8 dataset.name=coral model.load_in_4bit=true
#python train.py model.lm_model_name=google/gemma-2-9b-it model.whisper_model_name=openai/whisper-large-v3-turbo training.batch_size=16 dataset.name=coral model.load_in_4bit=true
#python train.py model.lm_model_name=google/gemma-2-9b-it model.whisper_model_name=openai/whisper-large-v3-turbo training.batch_size=16 dataset.name=coral model.load_in_4bit=true training.resume_checkpoint_path=/zhome/5c/8/186818/multimodal-alignment/outputs/2025-02-10/21-46-55/checkpoints/checkpoint_latest.pt training.resume_from_checkpoint=true training.kl_temp=1.0
python train.py model.lm_model_name=google/gemma-2-9b-it model.encoder_model_name=syvai/hviske-v2 training.batch_size=32 dataset.name=coral model.load_in_4bit=true wandb.enabled=true
