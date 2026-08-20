#!/usr/bin/env bash
# =============================================================================
# scripts/evaluate.sh
# Évalue le modèle fine-tuné sur le split test et applique le quality gate.
#
# Séquence :
#   1. evaluate.py → calcule WER/CER, produit metrics.json
#   2. quality_gate.py → vérifie les seuils, lève une erreur si dépassés
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Akilang Whisper — Évaluation"
echo "========================================"

cd "$PROJECT_ROOT"

echo "--- Étape 1/2 : Calcul des métriques ---"
python src/eval/evaluate.py ${HYDRA_OVERRIDES:-}

echo "--- Étape 2/2 : Quality Gate ---"
python src/eval/quality_gate.py ${HYDRA_OVERRIDES:-}

echo "✔ Évaluation et Quality Gate terminés"
