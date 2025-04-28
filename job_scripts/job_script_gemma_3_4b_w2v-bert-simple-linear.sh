#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J g3-w2v-bert-simple-linear
### Num cores
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
### Memory per core
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gemma_3_w2v_bert.out
#BSUB -eo gemma_3_w2v_bert.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate

python train.py model.lm_model_name=google/gemma-2-2b-it model.projector=simple model.encoder_model_name=facebook/w2v-bert-2.0 model.train_encoder_full=true training.batch_size=8 training.gradient_accumulation_steps=2 model.load_in_4bit=true wandb.enabled=true training.gradient_clip_norm=10.0 training.lr=1e-4 training.warmup_steps=500 model.downsample_factor=10