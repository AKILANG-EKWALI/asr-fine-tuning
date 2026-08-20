"""
src/eval/evaluate.py
Évaluation du modèle fine-tuné sur le split test.

Ce script :
  1. Charge le checkpoint final depuis training.output_dir/final
  2. Charge le split test (sans augmentation)
  3. Calcule WER et CER via le Seq2SeqTrainer en mode évaluation pure
  4. Sauvegarde les métriques dans metrics.json (utilisé par quality_gate.py et DVC)

Usage :
    python src/eval/evaluate.py
    python src/eval/evaluate.py training.output_dir=/path/to/outputs
"""

import json
import logging
import hydra
from omegaconf import DictConfig
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.load_data import (
    load_local_commonvoice_datasets,
    prepare_dataset_for_whisper,
    build_processor,
)
from src.data.collator import build_collator
from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Évalue le modèle final sur le split test et produit metrics.json.
    """

    # Chemin vers le checkpoint final sauvegardé après l'entraînement
    checkpoint_path = cfg.training.output_dir + "/final"
    logger.info(f"Chargement du checkpoint : {checkpoint_path}")

    # ==========================================================================
    # ÉTAPE 1 : Processor & Modèle
    # ==========================================================================

    processor = build_processor(cfg)

    # Chargement direct (sans LoRA wrapping) — le checkpoint contient les poids fusionnés
    # Si le modèle est un PeftModel, from_pretrained le détectera automatiquement
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

    # Fixe les tokens forcés pour contraindre la génération en Fulfuldé
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg.model.language,
        task=cfg.model.task,
    )
    model.config.suppress_tokens = []

    # ==========================================================================
    # ÉTAPE 2 : Chargement et préparation du split test
    # ==========================================================================

    logger.info("Chargement du dataset test...")
    datasets = load_local_commonvoice_datasets(cfg)

    # Colonnes à supprimer (gestion de l'absence de duration_column)
    cols_to_remove = [cfg.data.path_column, cfg.data.transcription_column, "audio"]
    if cfg.data.duration_column in datasets["test"].column_names:
        cols_to_remove.append(cfg.data.duration_column)

    logger.info("Préparation du split test (sans augmentation)...")
    test_dataset = datasets["test"].map(
        lambda ex: prepare_dataset_for_whisper(ex, processor, cfg),
        remove_columns=cols_to_remove,
        num_proc=cfg.training.dataloader_num_workers,
        desc="Feature extraction (test)",
    )

    collator = build_collator(processor, cfg)

    # ==========================================================================
    # ÉTAPE 3 : Configuration du Trainer en mode évaluation
    # ==========================================================================

    eval_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir + "/eval_tmp",  # Dossier temporaire
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        fp16=cfg.training.fp16,
        report_to=[],              # Pas de tracking pendant l'évaluation
        predict_with_generate=True,  # Génération déterministe (beam search)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    # ==========================================================================
    # ÉTAPE 4 : Évaluation et sauvegarde des métriques
    # ==========================================================================

    logger.info("Lancement de l'évaluation sur le split test...")
    # metric_key_prefix="test" → les clés seront "test_wer", "test_cer", etc.
    metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

    # Sauvegarde dans metrics.json (lu par quality_gate.py et tracké par DVC)
    metrics_path = "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    logger.info(f"Métriques sauvegardées dans {metrics_path}")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
