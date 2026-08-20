"""
src/data/load_data.py
Chargement et prétraitement du dataset Common Voice Fulfuldé local.

Ce module fournit trois fonctions principales :
  - filter_example   : filtre qualité sur durée et longueur de texte
  - load_local_commonvoice_datasets : lit les TSV locaux et construit un DatasetDict HF
  - prepare_dataset_for_whisper : extrait les features audio + les labels tokenisés
"""

import os
import logging
from datasets import load_dataset, Audio, DatasetDict
from omegaconf import DictConfig
from transformers import WhisperProcessor

# Logger module-level — les messages apparaissent dans la console et dans MLflow si configuré
logger = logging.getLogger(__name__)


# =============================================================================
# FILTRE QUALITÉ
# =============================================================================

def filter_example(example: dict, cfg: DictConfig) -> bool:
    """
    Retourne True si l'exemple est valide, False s'il doit être rejeté.

    Critères de rejet :
      1. Durée hors de [min_duration_sec, max_duration_sec] (si la colonne existe)
      2. Transcription vide ou nulle
      3. Transcription trop longue (> max_text_length mots)

    Args:
        example : une ligne du dataset (dict de colonnes)
        cfg     : configuration Hydra résolue

    Returns:
        bool
    """
    # --- Vérification de la durée (colonne optionnelle) ---
    dur_col = cfg.data.duration_column
    if dur_col in example and example[dur_col] is not None:
        duration = example[dur_col]
        if duration < cfg.data.min_duration_sec or duration > cfg.data.max_duration_sec:
            return False  # Audio trop court ou trop long

    # --- Vérification de la transcription ---
    text = example.get(cfg.data.transcription_column, "")
    if not text or len(text.strip()) == 0:
        return False  # Transcription absente ou vide

    # --- Vérification de la longueur du texte ---
    if len(text.split()) > cfg.data.max_text_length:
        return False  # Texte trop long pour le contexte de Whisper

    return True  # L'exemple passe tous les filtres


# =============================================================================
# CHARGEMENT DU DATASET
# =============================================================================

def load_local_commonvoice_datasets(cfg: DictConfig) -> DatasetDict:
    """
    Charge les splits train/validation/test depuis des fichiers TSV locaux.

    Étapes pour chaque split :
      1. Lecture du TSV avec load_dataset("csv", delimiter="\\t")
      2. Construction du chemin audio absolu
      3. Filtrage qualité via filter_example
      4. Cast de la colonne audio → format Audio(sampling_rate) (décodage automatique)
      5. Renommage en "audio" pour uniformité

    Args:
        cfg : configuration Hydra (cfg.data.*)

    Returns:
        DatasetDict avec les clés "train", "validation", "test"
    """
    # Chemins vers les fichiers TSV
    data_files = {
        "train":      os.path.join(cfg.data.metadata_dir, cfg.data.train_tsv),
        "validation": os.path.join(cfg.data.metadata_dir, cfg.data.eval_tsv),
        "test":       os.path.join(cfg.data.metadata_dir, cfg.data.test_tsv),
    }

    logger.info(f"Chargement des datasets depuis : {data_files}")

    # Lecture des TSV avec Hugging Face Datasets
    # delimiter="\t" car les fichiers Common Voice sont séparés par des tabulations
    datasets = load_dataset(
        "csv",
        data_files=data_files,
        delimiter="\t",
        cache_dir=cfg.data.cache_dir,
    )

    def add_audio_path(example: dict) -> dict:
        """
        Construit le chemin absolu vers le fichier audio.
        Certains TSV Common Voice préfixent le path avec "clips/" ; on le normalise.
        """
        relative_path = example[cfg.data.path_column]

        # Supprime le préfixe "clips/" si déjà présent dans le TSV
        if relative_path.startswith("clips/"):
            relative_path = relative_path[len("clips/"):]

        # Assemble le chemin complet vers le fichier audio
        example["audio_path"] = os.path.join(cfg.data.audio_dir, relative_path)
        return example

    for split in datasets:
        # Étape 1 : ajoute la colonne "audio_path" avec le chemin absolu
        datasets[split] = datasets[split].map(add_audio_path)

        # Étape 2 : filtre les exemples qui ne respectent pas les critères qualité
        datasets[split] = datasets[split].filter(
            lambda ex: filter_example(ex, cfg),
            num_proc=cfg.training.dataloader_num_workers,
        )

        # Étape 3 : cast de "audio_path" en colonne Audio
        #   → Hugging Face décodera automatiquement le fichier audio lors de l'accès
        #   → Le signal est resampleé à sample_rate si nécessaire
        datasets[split] = datasets[split].cast_column(
            "audio_path",
            Audio(sampling_rate=cfg.data.sample_rate),
        )

        # Étape 4 : renomme "audio_path" → "audio" (convention standard HF)
        datasets[split] = datasets[split].rename_column("audio_path", "audio")

        logger.info(f"Split '{split}' : {len(datasets[split])} exemples après filtrage")

    return datasets


# =============================================================================
# PRÉPARATION POUR WHISPER
# =============================================================================

def prepare_dataset_for_whisper(
    example: dict,
    processor: WhisperProcessor,
    cfg: DictConfig,
) -> dict:
    """
    Convertit un exemple brut en tenseurs prêts pour Whisper.

    - input_features : log-mel spectrogram 80 bandes (shape [80, 3000])
    - labels         : IDs des tokens de la transcription (avec tokens spéciaux)

    Args:
        example   : ligne du dataset contenant "audio" et la colonne texte
        processor : WhisperProcessor (feature extractor + tokenizer)
        cfg       : configuration Hydra

    Returns:
        dict avec "input_features" et "labels"
    """
    audio = example["audio"]

    # Extraction des features audio : log-mel spectrogram
    # Le processor normalise et padde/tronque à 30 secondes automatiquement
    input_features = processor(
        audio["array"],                    # Signal audio numpy float32
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",              # Retourne un tenseur PyTorch
    ).input_features[0]                   # [0] car le processor travaille en batch

    # Tokenisation de la transcription texte → liste d'IDs entiers
    labels = processor(text=example[cfg.data.transcription_column]).input_ids

    return {"input_features": input_features, "labels": labels}


# =============================================================================
# CONSTRUCTION DU PROCESSOR
# =============================================================================

def build_processor(cfg: DictConfig) -> WhisperProcessor:
    """
    Instancie le WhisperProcessor depuis Hugging Face Hub.

    Le processor combine :
      - WhisperFeatureExtractor : audio → log-mel spectrogram
      - WhisperTokenizer        : texte → token IDs (avec langue et tâche forcées)

    Args:
        cfg : configuration Hydra (cfg.model.*)

    Returns:
        WhisperProcessor configuré pour la langue et la tâche
    """
    logger.info(f"Chargement du processor : {cfg.model.name} | lang={cfg.model.language} | task={cfg.model.task}")
    return WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,  # "fuv" : Fulfuldé
        task=cfg.model.task,          # "transcribe"
    )
