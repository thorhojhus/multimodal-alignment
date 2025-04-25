import functools
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import load_checkpoint
from utils import generate_batch
from config import AppConfig
logger = logging.getLogger(__name__)

def make_plots(model: AudioLLMModel, cfg: AppConfig) -> None:
    # create dataset and processor
    if ("w2v-bert" in cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    elif ("BiCodec" in cfg.model.encoder_model_name):
        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    elif ("mimi" in cfg.model.encoder_model_name) or ("kyutai/mimi" == cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    else:
        try:
            processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
        except Exception:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    
    test_dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.test_split,
        tokenizer=model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
        config=cfg,
    )

    dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer, T=test_dataset.T), 
        shuffle=True,
    )
    
    load_checkpoint(
        [model.projector], None, None,
        "/work3/s224233/shared/outputs/2025-04-06/05-39-10/checkpoints/checkpoint_best.pt", logger,
        load_lr_from_checkpoint=cfg.training.load_lr_from_checkpoint
    )

    for batch in dataloader:
        # move data to GPU
        if isinstance(batch["audio_features"], dict):
            if "input_features" in batch["audio_features"]:
                # for wav2vec-bert/seamless models
                audio_features = {
                    'input_features': batch["audio_features"]["input_features"].to("cuda", non_blocking=True),
                    'attention_mask': batch["audio_features"]["attention_mask"].to("cuda", non_blocking=True)
                }
            elif "input_values" in batch["audio_features"]:
                # for Mimi models
                audio_features = {
                    'input_values': batch["audio_features"]["input_values"].to("cuda", non_blocking=True)
                }
            else:
                # fallback
                audio_features = {k: v.to("cuda", non_blocking=True) for k, v in batch["audio_features"].items()}
        else:
            # for other models (Whisper, CLAP, wav2vec)
            audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)

        text_ids = batch["text_ids"].to("cuda", non_blocking=True)
        text_ids = text_ids[0, 1 : 1 + (text_ids[0, 1:] != 0).sum()].unsqueeze(0)

        # Generation test:
        generated_texts, ground_truths, hidden_states, sequences = generate_batch(
            model, batch, cfg.model.lm_model_name, model.tokenizer, return_hidden_states=True
        )

        hidden_states_ = []
        for i in range(len(hidden_states[1:])):
            hidden_states_.append(hidden_states[i+1][-1][0].float())
        
        hidden_states_ = torch.stack(hidden_states_)

        with torch.no_grad():
            audio_embeddings = model._process_audio_features(audio_features)[0].float()
            text_embeddings: torch.Tensor = model.lm.get_input_embeddings()(text_ids)[0].float()

            # Aggregate:
            target_num = text_embeddings.shape[0]
            chunk_size = audio_embeddings.shape[0] // target_num

            audio_embeddings_pooled = []
            for i in range(target_num):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < target_num - 1 else audio_embeddings.shape[0]
                # Mean pooling over the chunk
                pooled_emb = torch.mean(audio_embeddings[start_idx:end_idx], dim=0)
                audio_embeddings_pooled.append(pooled_emb)
            audio_embeddings_pooled = torch.stack(audio_embeddings_pooled)
            # aggregate end
            
            # Override aggregation temporarily
            # audio_embeddings_pooled = audio_embeddings
            
            similarity_generated = torch.empty((audio_embeddings_pooled.shape[0], hidden_states_.shape[0]))
            for i in range(audio_embeddings_pooled.shape[0]):
                for j in range(hidden_states_.shape[0]):
                    similarity_generated[i, j] = torch.nn.functional.cosine_similarity(
                        audio_embeddings_pooled[i],
                        hidden_states_[j][0],
                        dim=0,
                    )
            # Make Heatmap plot with seaborn
            plt.figure(figsize=(12, 10))
            _ = sns.heatmap(
                similarity_generated,
                cmap="RdBu",
                linewidths=0.5,
                xticklabels=model.tokenizer.batch_decode(sequences[0], skip_special_tokens=True),
                cbar_kws={"label": "Cosine Similarity"},
            )
            plt.title('Cosine Similarity Between Audio and Text (Generated)')
            # Place the texts below the plot
            plt.figtext(0.1, -0.1, f"Generated Text: {generated_texts[0]}", wrap=True, fontsize=10, ha='left')
            plt.tight_layout()
            plt.savefig("/zhome/c2/2/146508/multimodal-alignment/similarity_figures/similarity_generated.png", dpi=300, bbox_inches='tight')

            similarity_gt = torch.empty((audio_embeddings_pooled.shape[0], text_embeddings.shape[0]))
            for i in range(audio_embeddings_pooled.shape[0]):
                for j in range(text_embeddings.shape[0]):
                    similarity_gt[i, j] = torch.nn.functional.cosine_similarity(
                        audio_embeddings_pooled[i].unsqueeze(0),
                        text_embeddings[j].unsqueeze(0),
                        dim=1,
                    )

            # Make Heatmap plot with seaborn
            plt.figure(figsize=(12, 10))
            _ = sns.heatmap(
                similarity_gt,
                cmap="RdBu",
                linewidths=0.5,
                xticklabels=model.tokenizer.batch_decode(text_ids[0]),
                cbar_kws={"label": "Cosine Similarity"},
            )
            plt.title('Cosine Similarity Between Audio and Text (Ground Truth)')
            # Place the texts below the plot
            plt.figtext(0.1, -0.2, f"Ground Truth Text: {ground_truths[0]}", wrap=True, fontsize=10, ha='left')
            plt.tight_layout()
            plt.savefig("/zhome/c2/2/146508/multimodal-alignment/similarity_figures/similarity_gt.png", dpi=300, bbox_inches='tight')
            return

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    # log the Hydra configuration for reproducibility
    logger.info("Hydra Config:\n" + str(cfg))
    # initialize components using config values from Hydra

    model = AudioLLMModel(cfg)
    make_plots(model, cfg)

if __name__ == "__main__":
    main()
