from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import WhisperProcessor
from omegaconf import DictConfig


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator custom pour Whisper Seq2Seq.
    Gère le padding indépendant du spectrogramme log-mel et des labels textuels.
    """
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Séparation des entrées audio et des labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # Padding des entrées (généralement déjà à 3000 frames)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Récupération des tokens de transcription
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Padding des labels à la longueur max du batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplacement du padding par -100 pour que la perte (CrossEntropyLoss) l'ignore
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Retrait du token de début de phrase s'il a déjà été ajouté lors de la tokenisation
        if len(labels) > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_collator(
    processor: WhisperProcessor,
    cfg: DictConfig,
) -> DataCollatorSpeechSeq2SeqWithPadding:
    """
    Construit le collator de données adapté à Whisper Seq2Seq.
    """
    return DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
