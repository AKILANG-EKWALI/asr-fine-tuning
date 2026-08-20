"""
src/model/lora_utils.py
Utilitaires pour l'application automatique de LoRA aux couches linéaires.

La détection automatique est utile lorsque l'architecture exacte du modèle
n'est pas connue à l'avance (ex. : lors d'un switch whisper-base → whisper-large).
"""

import torch.nn as nn
from typing import List


def find_all_linear_names(model: nn.Module) -> List[str]:
    """
    Détecte automatiquement tous les noms de couches nn.Linear dans le modèle.

    Cette fonction est utilisée quand cfg.model.lora.target_modules == ["auto"].
    Elle retourne les noms des modules linéaires (sauf "lm_head" qui est la tête
    de décodage — fine-tuner lm_head avec LoRA produit des instabilités).

    Algorithme :
      1. Parcourt tous les sous-modules nommés du modèle (récursivement)
      2. Sélectionne les nn.Linear
      3. Extrait le nom de feuille (dernier élément du chemin hiérarchique)
      4. Exclut "lm_head" de la liste finale

    Args:
        model : modèle PyTorch (ex. WhisperForConditionalGeneration)

    Returns:
        Liste de noms de modules linéaires compatibles avec LoRA

    Exemple de résultat :
        ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    """
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extrait le nom de la feuille (ex. "encoder.layers.0.self_attn.q_proj" → "q_proj")
            names = name.split(".")
            lora_module_names.add(names[-1] if len(names) > 1 else names[0])

    # "lm_head" est la couche de projection vers le vocabulaire — son fine-tuning
    # avec LoRA cause des problèmes de stabilité numérique
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    result = list(lora_module_names)
    return result
