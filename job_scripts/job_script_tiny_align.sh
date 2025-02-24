#!/bin/sh
### GPU
#BSUB -q gpua100
### Name
#BSUB -J tiny_align
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
#BSUB -oo tiny_align.out
#BSUB -eo tiny_align.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate

python train.py model.lm_model_name=google/gemma-2-2b-it model.encoder_model_name=facebook/wav2vec2-base-960h training.batch_size=32 model.tiny_align=true wandb.enabled=true training.num_epochs=100 training.save_every_n_steps=20000 training.evaluate_every_n_steps=10000 eval.full_eval=false