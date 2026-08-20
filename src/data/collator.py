"""
src/data/collator.py
Data collator pour le fine-tuning Whisper en mode Seq2Seq.

Le collator assemble un batch d'exemples hétérogènes (longueurs variables)
en tenseurs paddés prêts pour le forward pass du modèle.
"""

from transformers import DataCollatorSpeechSeq2SeqWithPadding, WhisperProcessor
from omegaconf import DictConfig


def build_collator(
    processor: WhisperProcessor,
    cfg: DictConfig,
) -> DataCollatorSpeechSeq2SeqWithPadding:
    """
    Construit le collator de données adapté à Whisper Seq2Seq.

    Ce collator gère deux champs :
      - "input_features" : log-mel spectrograms — paddés à la longueur max du batch
      - "labels"          : token IDs — paddés avec -100 (ignoré par CrossEntropyLoss)

    Args:
        processor : WhisperProcessor (contient le feature_extractor et le tokenizer)
        cfg       : configuration Hydra (actuellement non utilisée, pour extensibilité)

    Returns:
        DataCollatorSpeechSeq2SeqWithPadding configuré
    """
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        padding=True,             # Padde input_features et labels au max du batch
        pad_to_multiple_of=32,   # Padde à un multiple de 32 pour optimiser les ops CUDA (tensor cores)
        return_tensors="pt",     # Retourne des tenseurs PyTorch
    )
