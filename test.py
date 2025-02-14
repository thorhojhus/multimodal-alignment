from datetime import datetime
import logging
import torch.utils.data.dataloader
from torch.utils.data import DataLoader
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import hydra
from omegaconf import DictConfig
import torch
import jiwer
import wandb
from torchaudio.transforms import Resample

# set up a module-level logger
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="configs", config_name="config_eval")
def main(cfg: DictConfig):
    """ Test a third party model"""

    wandb.init(
        entity="borkeren",
        project="multimodal-alignment",
        name=f"test-model-whisper-{cfg.dataset.name}-{datetime.now()}",
        config=dict(cfg)
    )

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.dataset.name == "librispeech":
        from dataset.load_librispeech import librispeech_dataset_dict
        dataset = librispeech_dataset_dict["test.clean"]
    elif cfg.dataset.name == "coral":
        from dataset.load_coral import coral_dataset_dict
        dataset = coral_dataset_dict["test"]

    dataloader = DataLoader(dataset, batch_size=1)
    processor = WhisperProcessor.from_pretrained(cfg.model.third_party_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.third_party_model_id).to("cuda")

    need_resample = dataset[0]["audio"]["sampling_rate"] != processor.feature_extractor.sampling_rate
    if need_resample:
        print("Resampling is required")
        resampler = Resample(dataset[0]["audio"]["sampling_rate"], processor.feature_extractor.sampling_rate)
    else:
        print("Resampling is not required")
        
    
    WER_ERRORS = []

    for idx, batch in enumerate(dataloader):
        audio = batch["audio"]
        preprocessed_audio = resampler(audio["array"][0], resampler) if need_resample else audio["array"][0]
        input_features = processor(preprocessed_audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(batch['text'][0])
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        output = jiwer.compute_measures(reference, prediction)
        WER_ERRORS.append(output["wer"])
        if idx%100 == 0:
            logger.info(f"Iter: {idx} | WER: {output["wer"]:.2f}")
            logger.info(f"Expected: {reference}")
            logger.info(f"predicted: {prediction}")
            wandb.log({"WER": output["wer"]})
        
    logger.info(f"Average WER: {sum(WER_ERRORS) / len(WER_ERRORS)}")
    wandb.log({"Average WER": sum(WER_ERRORS) / len(WER_ERRORS)})
    

if __name__ == "__main__":
    main()