#!/usr/bin/env bash
# =============================================================================
# scripts/train.sh
# Lance le fine-tuning Whisper avec la configuration Hydra par défaut.
#
# Variables d'environnement utiles :
#   HYDRA_OVERRIDES : overrides Hydra supplémentaires
#     ex. : HYDRA_OVERRIDES="training.num_train_epochs=10" ./scripts/train.sh
# =============================================================================

set -euo pipefail  # Arrêt immédiat sur erreur, variable non définie, pipe failure

# Répertoire racine du projet (parent du répertoire scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Akilang Whisper — Fine-tuning"
echo " Répertoire projet : $PROJECT_ROOT"
echo "========================================"

cd "$PROJECT_ROOT"

# Lance l'entraînement avec Hydra
# Les overrides supplémentaires sont passés via la variable HYDRA_OVERRIDES
python src/training/train.py ${HYDRA_OVERRIDES:-}

echo "✔ Entraînement terminé"
