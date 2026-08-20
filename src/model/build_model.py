"""
src/model/build_model.py
Construction du modèle Whisper avec adaptateurs LoRA (via PEFT).

Deux modes :
  - use_peft=True  : adapte le modèle avec LoRA — seuls les adaptateurs sont entraînés
  - use_peft=False : fine-tuning complet du modèle (nécessite beaucoup plus de VRAM)
"""

import logging
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig

from src.model.lora_utils import find_all_linear_names

logger = logging.getLogger(__name__)


def build_model_with_lora(cfg: DictConfig) -> WhisperForConditionalGeneration:
    """
    Charge Whisper depuis le Hub et enrobe optionnellement avec LoRA.

    Étapes :
      1. Télécharge les poids pré-entraînés depuis HuggingFace Hub (ou cache local)
      2. Désactive forced_decoder_ids et suppress_tokens (ils seront réinitialisés
         au moment de l'entraînement pour forcer la langue cible)
      3. Si use_peft=True : applique LoRA sur les modules configurés
      4. Affiche le nombre de paramètres entraînables vs. total

    Args:
        cfg : configuration Hydra complète (cfg.model.*)

    Returns:
        Modèle Whisper prêt à l'entraînement (avec ou sans LoRA)
    """
    logger.info(f"Chargement du modèle : {cfg.model.name}")

    # --- Chargement du modèle pré-entraîné ---
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.name)

    # Reset des tokens forcés — ils seront fixés dynamiquement dans train.py
    # pour correspondre à la langue cible (Fulfuldé)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if not cfg.model.use_peft:
        # Mode fine-tuning complet : tous les poids sont entraînables
        logger.info("PEFT désactivé — fine-tuning complet du modèle")
        return model

    # --- Configuration LoRA ---
    # Détermination des modules cibles
    target_modules = list(cfg.model.lora.target_modules)  # conversion OmegaConf → list Python

    if not target_modules or target_modules == ["auto"]:
        # Détection automatique de tous les nn.Linear du modèle
        target_modules = find_all_linear_names(model)
        logger.info(f"Modules LoRA détectés automatiquement : {target_modules}")
    else:
        logger.info(f"Modules LoRA configurés : {target_modules}")

    lora_config = LoraConfig(
        r=cfg.model.lora.r,                       # Rang : contrôle la capacité des adaptateurs
        lora_alpha=cfg.model.lora.lora_alpha,     # Scaling : lora_alpha/r est le facteur effectif
        lora_dropout=cfg.model.lora.lora_dropout, # Dropout pour régularisation
        bias=cfg.model.lora.bias,                  # "none" = biais non entraîné
        target_modules=target_modules,            # Modules où insérer les adaptateurs
    )

    # --- Application de LoRA via PEFT ---
    model = get_peft_model(model, lora_config)

    # Affiche le ratio paramètres entraînables / total (typiquement 0.1–1%)
    model.print_trainable_parameters()

    return model
