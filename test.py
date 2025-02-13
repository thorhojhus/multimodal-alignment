import logging
import torch.utils.data.dataloader

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import hydra
from evaluate import load
from omegaconf import DictConfig
import torch
import jiwer

wer_metric = load("wer")
cer_metric = load("cer")

# set up a module-level logger
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="config_eval")
def main(cfg: DictConfig):
    """ Test a third party model"""

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if cfg.dataset.name == "librispeech":
        from dataset.load_librispeech import librispeech_dataset_dict
        dataset = librispeech_dataset_dict["test.clean"]
    elif cfg.dataset.name == "coral":
        from dataset.load_coral import coral_dataset_dict
        dataset = coral_dataset_dict["test"]


    processor = WhisperProcessor.from_pretrained(cfg.model.third_party_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.third_party_model_id).to("cuda")
    def map_to_pred(batch):
        breakpoint()
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        batch["reference"] = processor.tokenizer._normalize(batch['text'])
        with torch.no_grad():
            predicted_ids = model.generate(input_features.to("cuda"))[0]
        transcription = processor.decode(predicted_ids)
        batch["prediction"] = processor.tokenizer._normalize(transcription)
        return batch
    
    result = dataset.map(map_to_pred)
    output = jiwer.process_words(result["reference"], result["prediction"])
    logger.info(f"Done with WER: {output.wer:.2f}")
    

if __name__ == "__main__":
    main()