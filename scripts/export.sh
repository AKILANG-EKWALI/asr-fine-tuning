#!/usr/bin/env bash
# =============================================================================
# scripts/export.sh
# Fusionne les adaptateurs LoRA et exporte le modèle vers ONNX + CTranslate2.
#
# Prérequis :
#   - Le quality gate doit avoir été passé (metrics.json valide)
#   - optimum[onnxruntime] et ctranslate2 installés
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Akilang Whisper — Export du modèle"
echo "========================================"

cd "$PROJECT_ROOT"

python src/export/export_model.py ${HYDRA_OVERRIDES:-}

echo "✔ Export terminé"
