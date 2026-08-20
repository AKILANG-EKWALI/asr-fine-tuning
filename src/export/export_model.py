"""
src/export/export_model.py
Export du modèle fine-tuné vers les formats ONNX et CTranslate2 (Faster-Whisper).

Pipeline d'export :
  1. Chargement du checkpoint final (PeftModel ou modèle standard)
  2. Si PeftModel : fusion des adaptateurs LoRA dans les poids du modèle de base
     (merge_and_unload) et sauvegarde du modèle fusionné
  3. Export ONNX via optimum-cli (opset 14, tâche ASR)
  4. Conversion CTranslate2 via ct2-transformers-converter (quantization configurée)

Prérequis :
  - optimum[onnxruntime] installé (pour optimum-cli)
  - ctranslate2 installé (pour ct2-transformers-converter)

Usage :
    python src/export/export_model.py
    python src/export/export_model.py mlops.export.quantization=float16
"""

import subprocess
import logging
import sys
import hydra
from omegaconf import DictConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


def run_subprocess(cmd: list, step_name: str) -> None:
    """
    Lance une commande shell et lève une exception si elle échoue.

    Args:
        cmd       : liste de tokens de commande (ex. ["optimum-cli", "export", ...])
        step_name : nom de l'étape pour les messages d'erreur
    """
    logger.info(f"[{step_name}] Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(f"[{step_name}] ÉCHEC (code {result.returncode})")
        sys.exit(result.returncode)

    logger.info(f"[{step_name}] Succès ✔")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Exporte le modèle fine-tuné vers ONNX et CTranslate2.
    """

    checkpoint_path = cfg.training.output_dir + "/final"
    merged_path     = cfg.training.output_dir + "/merged"

    # ==========================================================================
    # ÉTAPE 1 : Fusion des adaptateurs LoRA (si applicable)
    # ==========================================================================

    logger.info(f"Chargement du checkpoint depuis : {checkpoint_path}")
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

    if hasattr(model, "merge_and_unload"):
        # Le modèle est un PeftModel (LoRA) → fusion des adaptateurs dans les poids
        # Après fusion, le modèle est un modèle Whisper standard exportable
        logger.info("Fusion des adaptateurs LoRA dans les poids du modèle de base...")
        model = model.merge_and_unload()

        # Sauvegarde du modèle fusionné (format HuggingFace standard)
        model.save_pretrained(merged_path)

        # Sauvegarde du processor (tokenizer + feature extractor) avec le modèle fusionné
        processor = WhisperProcessor.from_pretrained(checkpoint_path)
        processor.save_pretrained(merged_path)

        logger.info(f"Modèle fusionné sauvegardé dans : {merged_path}")
        source_path = merged_path  # L'export se fera à partir du modèle fusionné
    else:
        # Pas de LoRA : on exporte directement depuis le checkpoint
        logger.info("Pas d'adaptateurs LoRA détectés — export direct")
        source_path = checkpoint_path

    # ==========================================================================
    # ÉTAPE 2 : Export ONNX
    # ==========================================================================

    onnx_dir = cfg.mlops.export.onnx_dir
    logger.info(f"Export ONNX vers : {onnx_dir}")

    # optimum-cli export onnx convertit le modèle HF vers ONNX
    # --opset 14    : opset ONNX compatible avec onnxruntime >= 1.14
    # --task asr    : tâche de reconnaissance vocale (encoder + decoder)
    # --device cpu  : export sur CPU (pas de dépendance GPU pour l'export)
    run_subprocess([
        "optimum-cli", "export", "onnx",
        "--model", source_path,
        "--task", "automatic-speech-recognition",
        "--opset", "14",
        "--device", "cpu",
        onnx_dir,
    ], step_name="Export ONNX")

    logger.info(f"✔ Export ONNX terminé : {onnx_dir}")

    # ==========================================================================
    # ÉTAPE 3 : Conversion CTranslate2 (Faster-Whisper)
    # ==========================================================================

    ct2_dir = cfg.mlops.export.ct2_dir
    quantization = cfg.mlops.export.quantization  # ex. "int8_float16"

    logger.info(f"Conversion CTranslate2 vers : {ct2_dir} (quantization={quantization})")

    # ct2-transformers-converter convertit le modèle HF vers le format CTranslate2
    # --quantization int8_float16 : INT8 pour les matrices de poids, FP16 pour les activations
    # --force : écrase le répertoire de sortie s'il existe déjà
    run_subprocess([
        "ct2-transformers-converter",
        "--model", source_path,
        "--output_dir", ct2_dir,
        "--quantization", quantization,
        "--force",
    ], step_name="Export CTranslate2")

    logger.info(f"✔ Export CTranslate2 terminé : {ct2_dir}")

    print("\n=== Export complet ===")
    print(f"  ONNX       : {onnx_dir}")
    print(f"  CTranslate2: {ct2_dir}")


if __name__ == "__main__":
    main()
