#!/bin/sh
### GPU
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### Name
#BSUB -J 9b-w2v2-xlsr
### Num cores
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
### Memory per core
#BSUB -R "rusage[mem=4GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo latest_gemma_wav2vec_large.out
#BSUB -eo latest_gemma_wav2vec_large.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.8.0

cd /work3/s224233/shared/multimodal-alignment
source .venv/bin/activate

python train.py model.lm_model_name=google/gemma-2-9b-it model.encoder_model_name=facebook/wav2vec2-large-xlsr-53 model.projector=conv model.downsample_factor=6 training.batch_size=16 eval.batch_size=16 wandb.enabled=true dataset.name=\'coral,librispeech\'
