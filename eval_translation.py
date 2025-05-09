import os
import torch
import logging
import argparse
import torchaudio
import functools
from pathlib import Path
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict
from sacrebleu.metrics import BLEU, CHRF, TER
import hydra
from omegaconf import DictConfig
from transformers import BatchFeature

from comet import download_model, load_from_checkpoint

# Import the necessary components from the existing codebase
from config import AppConfig
from models.audio_llm import AudioLLMModel
from utils import load_checkpoint, generate_batch
from transformers import AutoProcessor, AutoFeatureExtractor
from dataset.audio_dataset import collate_fn

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fleurs_data():
    """Load and prepare FLEURS dataset for translation evaluation."""
    # Load the Danish and English FLEURS datasets
    fleurs_source = load_dataset("google/fleurs", "da_dk", split="test")
    #fleurs_source = load_dataset("google/fleurs", "da_dk", split="test")
    fleurs_target = load_dataset("google/fleurs", "en_us", split="test")
    
    # Group by ID to handle multiple speakers
    source_by_id = defaultdict(list)
    target_by_id = defaultdict(list)
    
    # Process Danish dataset
    for i, item in enumerate(fleurs_source):
        source_by_id[item['id']].append(i)
    
    # Process English dataset
    for i, item in enumerate(fleurs_target):
        target_by_id[item['id']].append(i)
    
    # Find common IDs
    common_ids = set(source_by_id.keys()).intersection(set(target_by_id.keys()))
    logger.info(f"Found {len(common_ids)} common IDs between Danish and English datasets")
    
    return fleurs_source, fleurs_target, source_by_id, target_by_id, common_ids

def _format_translation_chat(tokenizer, model_name):
    """Create a translation prompt template based on model type."""
    if "gemma-2" in model_name.lower() or "gemma-3" in model_name.lower():
        # For Gemma models
        return [
            {"role": "user", "content": f"You are a helpful translator that provides accurate translations. Translate the audio accurately and provide only the translation. Remember to output in English. \n\nTranslate the following audio from Danish to English:\nDanish audio: <|audio|> \nEnglish translation: "},
        ]
    else:
        # For other models (default)
        return [
            {"role": "system", "content": "You are a helpful translator. Translate the audio input to English accurately. Provide only the translation."},
            {"role": "user", "content": f"Translate the following audio to English: <|audio|>"},
            {"role": "assistant", "content": ""}
        ]

def process_audio_features(audio_array, audio_sampling_rate, processor):
    """Process audio features according to encoder type."""
    target_sampling_rate = 16000
    
    # Resample if needed
    if audio_sampling_rate != target_sampling_rate:
        audio_array = torchaudio.functional.resample(
            audio_array, 
            orig_freq=audio_sampling_rate, 
            new_freq=target_sampling_rate
        )
    
    # Process according to processor type
    if "Wav2Vec2FeatureExtractor" in processor.__class__.__name__ or "w2v-bert" in str(processor.__class__):
        if (len(audio_array) / target_sampling_rate) > 16:  # truncate audio to 16 seconds
            audio_array = audio_array[:target_sampling_rate*16]
        
        # For W2V-BERT, return features in BatchFeature format
        features = processor(
            audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt"
        )
        
        # Check if we have the right format
        if hasattr(features, 'input_features'):
            return features
        else:
            # Create appropriate BatchFeature format
            features_tensor = features.input_values if hasattr(features, 'input_values') else features
            attention_mask = torch.ones(features_tensor.shape[1] if len(features_tensor.shape) > 1 else features_tensor.shape[0]).unsqueeze(0)
            return BatchFeature({
                'input_features': features_tensor,
                'attention_mask': attention_mask
            })
    
    elif "SeamlessM4TFeatureExtractor" in processor.__class__.__name__:
        # Dont filter when testing only for training
        # if (len(audio_array) / target_sampling_rate) > 16:  # truncate audio to 16 seconds
        #     audio_array = audio_array[:target_sampling_rate*16]
        #     pass
        return processor(
            audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt",
        )
    
    elif "Whisper" in processor.__class__.__name__:
        return processor(
            audio=audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt",
            truncation=True,
        ).input_features[0]
    
    elif "MimiFeatureExtractor" in processor.__class__.__name__ or "Mimi" in processor.__class__.__name__:
        # Normalize audio to prevent clipping
        audio_array = audio_array / (audio_array.abs().max() + 1e-8)
        return processor(
            raw_audio=audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt"
        )
    
    else:
        # Default processing
        return processor(
            audio=audio_array,
            sampling_rate=target_sampling_rate,
            return_tensors="pt"
        ).input_features[0]

def prepare_translation_samples(model, dataset, indices, processor, tokenizer):
    """Prepare samples for translation evaluation."""
    samples = []
    
    for idx in indices:
        item = dataset[idx]
        
        # Get audio data
        audio_array = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        audio_sampling_rate = item["audio"]["sampling_rate"]
        
        # Process audio features
        audio_features = process_audio_features(audio_array, audio_sampling_rate, processor)
        
        # Format conversation with translation prompt
        conversation = _format_translation_chat(tokenizer, model.config.model.lm_model_name)
        
        # Tokenize conversation
        tokenized = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt"
        )[0]
        
        samples.append({
            "input_ids": tokenized,
            "alt_input_ids": tokenized,
            "audio_features": audio_features,
            "raw_audio": audio_array,
            "labels": tokenized,
            "alt_labels": tokenized,
            "raw_audio": audio_array,
            "text_ids": tokenized,
        })
    
    return samples

def evaluate_translations(hypotheses, references, sources=None):
    """Evaluate translations using sacrebleu and COMET (if available)."""
    # Calculate BLEU score with sacrebleu
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    
    # Calculate CHRF score
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, [references])
    
    # Calculate TER score
    ter = TER()
    ter_score = ter.corpus_score(hypotheses, [references])
    
    results = {
        "BLEU": bleu_score.score,
        "CHRF": chrf_score.score,
        "TER": ter_score.score,
    }
    
    # Use COMET if available and source texts are provided
    # Initialize COMET model
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    
    # Prepare data for COMET
    data = [
        {
            "src": src,
            "mt": hyp,
            "ref": ref
        }
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]
    
    # Get COMET scores
    comet_scores = comet_model.predict(data, batch_size=8, gpus=1)
            
    # Extract the score
    results["COMET"] = comet_scores["system_score"]


    
    return results

@hydra.main(version_base="1.2", config_path="configs", config_name="config_eval_wav2vec")
def main(hydra_conf: DictConfig):
    parser = argparse.ArgumentParser(description="Evaluate audio translation using FLEURS dataset")
    parser.add_argument("--config_path", type=str, required=False, help="Path to the model config file")
    parser.add_argument("--checkpoint_path", type=str, default="/zhome/5c/8/186818/shared/outputs/2025-04-25/06-42-49/checkpoints/checkpoint_latest.pt", help="Path to the checkpoint file")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    args = parser.parse_args()
    
    # Load configuration
    app_config = AppConfig(**hydra_conf)
    
    # Update config with checkpoint path
    app_config.training.resume_checkpoint_path = args.checkpoint_path
    
    # Initialize model
    model = AudioLLMModel(app_config)
    model = model.to("cuda")
    model.eval()
    
    # Load checkpoint
    checkpoint_path = app_config.training.resume_checkpoint_path
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    load_components = [model.projector]
    if app_config.model.train_encoder_full:
        load_components.append(model.audio_encoder.audio_encoder)
    
    global_step, _ = load_checkpoint(
        load_components, None, None,
        checkpoint_path, logger,
        load_lr_from_checkpoint=False
    )
    
    # Initialize processor based on model type
    if "w2v-bert" in app_config.model.encoder_model_name:
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(app_config.model.encoder_model_name, padding_value=0.0)
    elif "BiCodec" in app_config.model.encoder_model_name:
        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    elif "mimi" in app_config.model.encoder_model_name.lower():
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    else:
        try:
            processor = AutoProcessor.from_pretrained(app_config.model.encoder_model_name)
        except Exception:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(app_config.model.encoder_model_name)
    
    # Load FLEURS datasets
    fleurs_da, fleurs_en, source_by_id, target_by_id, common_ids = load_fleurs_data()
    
    # Translate Danish to English
    all_translations = []
    all_references = []
    all_source_texts = []  # Keep Danish source texts for reference
    
    # Process in batches of pairs
    common_ids_list = list(common_ids)[:args.num_samples]
    
    for batch_start in tqdm(range(0, len(common_ids_list), args.batch_size)):
        batch_ids = common_ids_list[batch_start:batch_start+args.batch_size]
        
        for id in batch_ids:
            # Get Danish samples for this ID
            source_indices = source_by_id[id]
            
            # Get English reference for this ID
            target_idx = target_by_id[id][0]  # Use first one as reference
            reference = fleurs_en[target_idx]["raw_transcription"]
            
            # Prepare translation samples
            samples = prepare_translation_samples(model, fleurs_da, source_indices, processor, model.tokenizer)
            
            # Process each sample (one by one for stability)
            for i, sample in enumerate(samples):
                # Create a batch with just this sample
                batch = collate_fn([sample], model.tokenizer, T=app_config.model.token_length if hasattr(app_config.model, 'token_length') else 30)
                
                # Generate translation
                with torch.inference_mode():
                    generated_texts, _ = generate_batch(
                        model, batch, app_config.model.lm_model_name, model.tokenizer,
                        max_new_tokens=100, do_sample=True
                    )
                
                # Add translation and reference
                translation = generated_texts[0]
                all_translations.append(translation)
                all_references.append(reference)
                all_source_texts.append(fleurs_da[source_indices[i]]["raw_transcription"])
                # Clean up any extra tokens or formatting that might be present
                translation = translation.strip()
                
                # Log sample to console for monitoring
                logger.info(f"Sample {len(all_translations)}:")
                logger.info(f"Source (Danish): {fleurs_da[source_indices[i]]['raw_transcription']}")
                logger.info(f"Reference (English): {reference}")
                logger.info(f"Translation: {translation}")
                logger.info("-" * 50)
    
    # Calculate evaluation metrics
    logger.info("\nCalculating translation metrics...")
    
    # Ensure we have equal numbers of translations and references
    assert len(all_translations) == len(all_references), "Number of translations and references must match"
    
    # Calculate and print metrics
    metrics = evaluate_translations(all_translations, all_references, all_source_texts)
    
    logger.info("\nTranslation Evaluation Results:")
    logger.info(f"Number of samples evaluated: {len(all_translations)}")
    logger.info(f"BLEU score: {metrics['BLEU']:.4f}")
    logger.info(f"CHRF score: {metrics['CHRF']:.4f}")
    logger.info(f"TER score: {metrics['TER']:.4f}")
    if "COMET" in metrics:
        logger.info(f"COMET score: {metrics['COMET']:.4f}")
    
    # Save results to file
    output_dir = Path("translation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "translation_details.txt", "w", encoding="utf-8") as f:
        f.write(f"Model checkpoint: {args.checkpoint_path}\n")
        f.write(f"Number of samples: {len(all_translations)}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"BLEU score: {metrics['BLEU']:.4f}\n")
        f.write(f"CHRF score: {metrics['CHRF']:.4f}\n")
        f.write(f"TER score: {metrics['TER']:.4f}\n")
        if "COMET" in metrics:
            f.write(f"COMET score: {metrics['COMET']:.4f}\n")
        f.write("\n")
        
        f.write("Sample Translations:\n")
        for i in range(min(10, len(all_translations))):  # Save first 10 examples
            f.write(f"Sample {i+1}:\n")
            f.write(f"Source (Danish): {all_source_texts[i]}\n")
            f.write(f"Reference (English): {all_references[i]}\n")
            f.write(f"Translation: {all_translations[i]}\n")
            f.write("-" * 50 + "\n")
    
    # Save all translations and references in CSV format for further analysis
    import csv
    with open(output_dir / "all_translations.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source (Danish)", "Reference (English)", "Translation"])
        for src, ref, trans in zip(all_source_texts, all_references, all_translations):
            writer.writerow([src, ref, trans])
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()