def template(name: str, config: str, output_file_id: str) -> str:
    return f"""#!/bin/sh
### GPU
#BSUB -q gpua100
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
#BSUB -oo generated_job_scripts_{name}/runlogs/{output_file_id}.out
#BSUB -e ~/multimodal-alignment/multimodal-alignment/job_outs/gpu_%J.err
#BSUB -eo generated_job_scripts_{name}/runlogs/{output_file_id}.err
# -- end of LSF options --

nvidia-smi

module swap python3/3.12.7
module swap cuda/12.6.3

cd ~/multimodal-alignment
source .venv/bin/activate
python train.py {config}"""


def parse_csv(file_path) -> list[dict[str, str | int]]:
    """
    Parses a CSV file and returns a list of dictionaries.
    Columns are: [LLM MODEL, ENCODER MODEL, PROJECTION, params, DATASET(s), RUNCONFIG, RUNNAME]
    Only really uses the first 8, as the last 2 are extra.

    parses LLM model to the string              'model.lm_model_name=LLM MODEL'
    parses ENCODER MODEL to the string          'model.encoder_model_name=ENCODER MODEL'
    parses PROJECTION to the string             'model.projector=PROJECTION' (based on map)
    parses params to the string                 'model.PARAMS[0]=PARAMS[1]' (split on =)
    parses the datasets to the string           'dataset.name=DATASET(s)' (split on ,)
    parses the runconfig to the list of strings [f'{pair[0]}={pair[1]}' for pair in split(RUNCONFIG, ',')]
    """
    import csv
    
    gemma_map: dict = {
        "Gemma 2 9B": "google/gemma-2-9b-it",
        "Gemma 2 2B": "google/gemma-2-2b-it"
    }

    # Define the projection map
    projection_map: dict[str, str] = {
        "Linear": "linear",
        "Ultravox Style": "simple",
        "Audio Codec": None,
        "TinyAlign": "tinyalign",
        "CNN": "conv"
    }

    encoder_map: dict[str, str] = {
        "Whisper Large": "openai/whisper-large-v3",
        "Hviske V2": "syvai/hviske-v2",
        "Wav2Vec2 Base": "facebook/wav2vec2-base-960h",
        "Wav2Vec2 Large": "facebook/wav2vec2-large",
        "Wav2Vec2 Bert": "facebook/w2v-bert-2.0"
    }
    
    result = []
    
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)  # Skip header row
        
        for id, row in enumerate(reader):
            # Create a dictionary for each row
            config_dict = {}
            
            # Parse LLM MODEL
            config_dict['model.lm_model_name'] = gemma_map.get(row[0].strip())
            
            # Parse ENCODER MODEL using the encoder map
            encoder_value = row[1].strip()
            mapped_encoder = encoder_map.get(encoder_value, encoder_value)  # Use original if not in map
            config_dict['model.encoder_model_name'] = mapped_encoder
            
            # Parse PROJECTION using the projection map
            projection_value = row[2].strip()
            mapped_projection = projection_map.get(projection_value)
            if mapped_projection is not None:  # Regular mapped projections
                config_dict['model.projector'] = mapped_projection
            elif projection_value == "Audio Codec":  # Special case for Audio Codec
                config_dict['model.audio_tokenizer'] = "true"
            
            # Parse params (assuming params is in format "key=value")
            if '=' in row[3]:
                param_key, param_value = row[3].split('=', 1)
                config_dict[f'model.{param_key}'] = int(param_value)
            
            # Parse DATASET(s) (split by comma if multiple)
            datasets = row[4].split(',')
            if len(datasets) == 1:
                config_dict['dataset.name'] = datasets[0].strip().lower()
            else:
                config_dict['dataset.name'] = [ds.strip() for ds in datasets]
            
            # Parse RUNCONFIG (split by comma and create key-value pairs)
            if row[5]:  # Check if RUNCONFIG is not empty
                runconfig_pairs = row[5].split(',')
                for pair in runconfig_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        config_dict[key.strip()] = value.strip()
            
            # Add RUNNAME
            config_dict['wandb.run_name'] = f"\"{id} - {row[6].replace("(", "\\(").replace(")", "\\)")}\""
            config_dict['id'] = id
            result.append(config_dict)
    
    return result

def csv_dict_to_config_string(config_dict: dict[str, str | int | list]) -> tuple[str, int]:
    """Coverts the output of the config dict to a string that can be used in the config file."""
    output_string = ""
    for key, value in config_dict.items():
        if key == "id":
            continue
        if isinstance(value, list):
            value = ','.join(value)
        output_string += f"{key}={value} "
    return output_string.strip()


def write_job_template(name: str, config: str, output_id: int):
    """Writes the job template to a file."""
    with open(f"generated_job_scripts_{name}/job_{output_id}.sh", "w") as f:
        f.write(template(name, config, output_id))

if __name__ == "__main__":
    # get args
    import argparse
    parser = argparse.ArgumentParser(description="Parse a CSV file and generate job scripts.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")

    # Parse the arguments
    args = parser.parse_args()
    # Call the function with the provided CSV file
    pdata = parse_csv(args.csv_file)
    for i, pdict in enumerate(pdata):
        if i % 2 == 0:
            write_job_template("oliver", csv_dict_to_config_string(pdict), pdict['id'])
        else:
            write_job_template("thor", csv_dict_to_config_string(pdict), pdict['id'])