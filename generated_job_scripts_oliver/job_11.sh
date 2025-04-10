#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
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
#BSUB -oo generated_job_scripts_oliver/runlogs/4.out
#BSUB -e ~/multimodal-alignment/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo generated_job_scripts_oliver/runlogs/4.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.6.3

cd ~/multimodal-alignment
source .venv/bin/activate
python train.py model.lm_model_name=google/gemma-2-9b-it model.encoder_model_name=syvai/hviske-v2 model.audio_tokenizer=true model.audio_tokenizer_vocab_size=8192 dataset.name=coral wandb.enabled=true +wandb.project=benchmarks model.load_in_4bit=true wandb.run_name="4 - Gemma 2 9B + Hviske V2 + Audio Codec \(Coral\) w. Noise" dataset.augment=gaussian