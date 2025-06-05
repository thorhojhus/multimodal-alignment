#!/bin/sh
### GPU
#BSUB -q gpua100
### Name
#BSUB -J coral_bench
### Num cores
#BSUB -n 8
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
#BSUB -oo coral_bench.out
#BSUB -eo coral_bench.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate

python test.py model.lm_model_name=google/gemma-3-12b-it model.projector=simple model.encoder_model_name=facebook/w2v-bert-2.0 model.train_encoder_full=true training.batch_size=8 training.gradient_accumulation_steps=4 model.load_in_4bit=false wandb.enabled=true training.gradient_clip_norm=1.0 training.lr=1e-4 training.warmup_steps=500 model.downsample_factor=8 dataset.name=coral training.num_epochs=2 training.resume_checkpoint_path="/work3/s224233/shared/outputs/2025-05-29/10-29-20/checkpoints/checkpoint_latest.pt" training.resume_from_checkpoint=true eval.batch_size=32