#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J 2b_w2v
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
#BSUB -oo gemma2b_w2v.out
#BSUB -eo gemma2b_w2v.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate

python train.py model.lm_model_name=google/gemma-2-2b-it model.encoder_model_name=facebook/w2v-bert-2.0 model.train_encoder_full=true model.projector=simple model.downsample_factor=8 dataset.name=coral wandb.enabled=true +wandb.project=benchmarks model.load_in_4bit=true "wandb.run_name=Gemma 2 2B + w2v-bert + Ultravox Style \(Coral\)" training.batch_size=32 training.gradient_clip_norm=1.0 training.lr=1e-4 training.warmup_steps=500 training.num_epochs=3


python train.py model.lm_model_name=google/gemma-2-2b-it model.encoder_model_name=facebook/w2v-bert-2.0 model.train_encoder_full=true model.projector=simple model.downsample_factor=8 dataset.name=coral wandb.enabled=true model.load_in_4bit=true "wandb.run_name=Gemma 2 2B + w2v-bert + Ultravox Style \(Coral\)" training.batch_size=8 training.gradient_clip_norm=1.0 training.lr=3e-4 training.warmup_steps=500 training.num_epochs=1