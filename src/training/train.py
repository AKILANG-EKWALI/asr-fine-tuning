"""
src/training/train.py
Point d'entrée principal pour le fine-tuning Whisper sur le Fulfuldé.

Orchestration :
  1. Hydra charge et résout toute la configuration
  2. Le processor et le modèle (avec LoRA) sont instanciés
  3. Le dataset local est chargé, filtré, augmenté, et tokenisé
  4. MLflow ouvre un run et log tous les hyperparamètres
  5. Le Seq2SeqTrainer HuggingFace lance l'entraînement
  6. Le meilleur modèle et le processor sont sauvegardés

Usage :
    python src/training/train.py                           # Config par défaut
    python src/training/train.py training.num_train_epochs=10  # Override Hydra
    python src/training/train.py model=whisper_small_lora  # Autre config modèle
"""

import logging
import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.load_data import (
    load_local_commonvoice_datasets,
    prepare_dataset_for_whisper,
    build_processor,
)
from src.data.augment import TelephoneAugmenter
from src.data.collator import build_collator
from src.model.build_model import build_model_with_lora
from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Fonction principale d'entraînement, décorée par @hydra.main.

    Le décorateur @hydra.main :
      - Résout toutes les interpolations de cfg (${...})
      - Change le répertoire courant vers hydra.run.dir
      - Initialise le logging Hydra
    """

    # --- Log de la configuration complète résolue ---
    logger.info("Configuration complète :\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # --- Reproductibilité ---
    torch.manual_seed(cfg.seed)

    # ==========================================================================
    # ÉTAPE 1 : Processor & Modèle
    # ==========================================================================

    logger.info("Initialisation du processor Whisper...")
    processor = build_processor(cfg)

    logger.info("Construction du modèle avec LoRA...")
    model = build_model_with_lora(cfg)

    # Fixe les tokens forcés au décodeur pour contraindre la langue (Fulfuldé)
    # et la tâche (transcription) — crucial pour ne pas transcriber en anglais
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=cfg.model.language,
        task=cfg.model.task,
    )

    # ==========================================================================
    # ÉTAPE 2 : Chargement et préparation du dataset
    # ==========================================================================

    logger.info("Chargement des datasets locaux...")
    datasets = load_local_commonvoice_datasets(cfg)

    # Instanciation de l'augmenteur téléphonique (réutilisé pour chaque exemple train)
    augmenter = TelephoneAugmenter(cfg)

    def augment_and_prepare(example: dict) -> dict:
        """
        Combine augmentation + préparation Whisper pour le split train.
        L'augmentation est aléatoire (prob=0.7 par défaut).
        """
        audio = example["audio"]
        # Augmentation du signal brut avant extraction de features
        audio["array"] = augmenter(audio["array"], audio["sampling_rate"])
        return prepare_dataset_for_whisper(example, processor, cfg)

    # Colonnes à supprimer après la transformation (elles ne sont plus nécessaires)
    # Note : on gère le cas où duration_column est absent du TSV
    cols_to_remove = [
        cfg.data.path_column,
        cfg.data.transcription_column,
        "audio",
    ]
    # Ajoute duration_column seulement si elle existe dans le dataset
    if cfg.data.duration_column in datasets["train"].column_names:
        cols_to_remove.append(cfg.data.duration_column)

    logger.info("Préparation du split train (avec augmentation)...")
    train_dataset = datasets["train"].map(
        augment_and_prepare,
        remove_columns=cols_to_remove,
        num_proc=cfg.training.dataloader_num_workers,
        desc="Augmentation + feature extraction (train)",
    )

    # Colonnes à supprimer pour la validation (mêmes colonnes, sans duration si absente)
    eval_cols_to_remove = [c for c in cols_to_remove if c in datasets["validation"].column_names]

    logger.info("Préparation du split validation (sans augmentation)...")
    eval_dataset = datasets["validation"].map(
        lambda ex: prepare_dataset_for_whisper(ex, processor, cfg),
        remove_columns=eval_cols_to_remove,
        num_proc=cfg.training.dataloader_num_workers,
        desc="Feature extraction (validation)",
    )

    # ==========================================================================
    # ÉTAPE 3 : Collator & Arguments d'entraînement
    # ==========================================================================

    collator = build_collator(processor, cfg)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        save_total_limit=cfg.training.save_total_limit,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        remove_unused_columns=cfg.training.remove_unused_columns,
        label_names=list(cfg.training.label_names),  # conversion OmegaConf → list Python
        # Rapporte vers MLflow si c'est le backend configuré
        report_to=["mlflow"] if cfg.mlops.tracking_backend == "mlflow" else [],
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.seed,
        predict_with_generate=True,   # Active la génération pour le calcul WER/CER
    )

    # ==========================================================================
    # ÉTAPE 4 : MLflow — suivi de l'expérience
    # ==========================================================================

    mlflow.set_tracking_uri(cfg.mlops.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlops.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlops.mlflow.run_name):

        # Log de tous les hyperparamètres résolus (config complète aplatie)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # ======================================================================
        # ÉTAPE 5 : Entraînement
        # ======================================================================

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            # Fonction de métriques : WER et CER normalisés
            compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
            tokenizer=processor.feature_extractor,  # Utilisé pour le padding audio
        )

        logger.info("Lancement de l'entraînement...")
        trainer.train()

        # ======================================================================
        # ÉTAPE 6 : Sauvegarde du meilleur modèle
        # ======================================================================

        final_dir = cfg.training.output_dir + "/final"
        logger.info(f"Sauvegarde du modèle final dans : {final_dir}")
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)

        # Log de la métrique finale dans MLflow
        best_wer = trainer.state.best_metric
        mlflow.log_metric("best_wer", best_wer if best_wer is not None else -1)
        logger.info(f"Entraînement terminé. Meilleur WER : {best_wer}")


if __name__ == "__main__":
    main()
