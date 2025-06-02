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
#BSUB -oo generated_job_scripts_oliver/runlogs/17.out
#BSUB -e ~/multimodal-alignment/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo generated_job_scripts_oliver/runlogs/17.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.6.3

cd /work3/s224233/shared/rundir
source .venv/bin/activate
python train.py model.lm_model_name=google/gemma-3-12b-it model.encoder_model_name=openai/whisper-large-v3 model.projector=simple model.downsample_factor=2 dataset.name=coral wandb.enabled=true +wandb.project=benchmarks model.load_in_4bit=true wandb.run_name="17 - Gemma 3 12B + Whisper Large + Ultravox Style \(Coral\)" training.batch_size=2