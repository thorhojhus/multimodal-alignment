#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J 9b_tok_whi
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
#BSUB -oo latest_gemma_audio_tokenizer_whisper.out
#BSUB -eo latest_gemma_audio_tokenizer_whisper.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate


python train.py model.lm_model_name=google/gemma-2-9b-it model.encoder_model_name=openai/whisper-large-v3 training.batch_size=32 eval.batch_size=32 dataset.name=coral model.load_in_4bit=true model.audio_tokenizer=true training.kl_loss=true wandb.enabled=true training.num_epochs=5