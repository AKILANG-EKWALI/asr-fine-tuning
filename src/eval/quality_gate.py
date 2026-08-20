"""
src/eval/quality_gate.py
Quality Gate : valide les métriques avant d'autoriser l'export du modèle.

Ce script lit metrics.json (produit par evaluate.py) et compare le WER et CER
aux seuils configurés dans cfg.mlops.quality_gate.

Comportement :
  - Si les seuils sont respectés → affiche "Quality Gate PASSÉ ✔" et retourne 0
  - Si un seuil est dépassé      → lève RuntimeError et retourne un code non-nul
    (ce qui fait échouer le pipeline DVC / la CI)

Usage :
    python src/eval/quality_gate.py
    python src/eval/quality_gate.py mlops.quality_gate.max_wer=20.0
"""

import json
import sys
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Lit metrics.json et vérifie le respect des seuils WER/CER configurés.
    """

    # --- Lecture des métriques produites par evaluate.py ---
    metrics_path = "metrics.json"
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        logger.error(f"Fichier {metrics_path} introuvable — exécutez d'abord evaluate.py")
        sys.exit(1)

    # --- Récupération des valeurs WER et CER ---
    # On utilise 100.0 comme valeur par défaut si la clé est absente
    # (cas où l'évaluation n'a pas calculé la métrique)
    test_wer = metrics.get("test_wer", 100.0)
    test_cer = metrics.get("test_cer", 100.0)

    # --- Récupération des seuils depuis la configuration ---
    max_wer = cfg.mlops.quality_gate.max_wer  # ex. 25.0 %
    max_cer = cfg.mlops.quality_gate.max_cer  # ex. 30.0 %

    # --- Affichage des résultats ---
    print("=" * 50)
    print("QUALITY GATE — Résultats")
    print("=" * 50)
    print(f"  WER test : {test_wer:.2f}%  (seuil max : {max_wer}%)")
    print(f"  CER test : {test_cer:.2f}%  (seuil max : {max_cer}%)")
    print("=" * 50)

    # --- Vérification des conditions de passage ---
    failures = []

    if test_wer > max_wer:
        failures.append(f"WER={test_wer:.2f}% > {max_wer}%")

    if test_cer > max_cer:
        failures.append(f"CER={test_cer:.2f}% > {max_cer}%")

    if failures:
        # Le gate échoue : le pipeline DVC / CI sera arrêté
        msg = "Quality Gate ÉCHOUÉ : " + ", ".join(failures)
        logger.error(msg)
        raise RuntimeError(msg)  # Code de retour non-nul pour DVC et les scripts shell

    # Tout est dans les clous → l'export peut commencer
    print("✔  Quality Gate PASSÉ — Export autorisé")
    logger.info("Quality Gate passé avec succès")


if __name__ == "__main__":
    main()
