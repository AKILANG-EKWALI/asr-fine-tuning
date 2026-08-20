# =============================================================================
# Makefile
# Raccourcis pour les tâches courantes du projet Akilang Whisper.
#
# Usage :
#   make train         → Lance le fine-tuning
#   make evaluate      → Évalue le modèle + quality gate
#   make export        → Exporte le modèle (ONNX + CTranslate2)
#   make test          → Lance la suite de tests unitaires
#   make docker-train  → Lance l'entraînement dans Docker
#   make dvc-repro     → Reproduit le pipeline DVC complet
#   make lint          → Vérification du style de code
#   make clean         → Nettoie les fichiers temporaires
# =============================================================================

.PHONY: train evaluate export test docker-train docker-evaluate docker-export \
        dvc-repro lint clean help

# Commande Python (peut être overridée : make train PYTHON=python3.10)
PYTHON ?= python

# Overrides Hydra optionnels (ex. : make train HYDRA_OVERRIDES="training.num_train_epochs=10")
HYDRA_OVERRIDES ?=

# Cible par défaut
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
train:
	@echo "==> Lancement du fine-tuning Whisper..."
	$(PYTHON) src/training/train.py $(HYDRA_OVERRIDES)

# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
evaluate:
	@echo "==> Évaluation sur le split test..."
	$(PYTHON) src/eval/evaluate.py $(HYDRA_OVERRIDES)
	@echo "==> Quality Gate..."
	$(PYTHON) src/eval/quality_gate.py $(HYDRA_OVERRIDES)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
export:
	@echo "==> Export ONNX + CTranslate2..."
	$(PYTHON) src/export/export_model.py $(HYDRA_OVERRIDES)

# ---------------------------------------------------------------------------
# Pipeline complet (train → evaluate → quality_gate → export)
# ---------------------------------------------------------------------------
pipeline: train evaluate export
	@echo "==> Pipeline complet terminé ✔"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-train:
	@echo "==> Entraînement dans Docker (GPU)..."
	docker compose run --rm train $(HYDRA_OVERRIDES)

docker-evaluate:
	@echo "==> Évaluation dans Docker..."
	docker compose run --rm evaluate

docker-export:
	@echo "==> Export dans Docker..."
	docker compose run --rm export

docker-build:
	@echo "==> Build des images Docker..."
	docker compose build

# ---------------------------------------------------------------------------
# DVC
# ---------------------------------------------------------------------------
dvc-repro:
	@echo "==> Reproduction du pipeline DVC..."
	dvc repro

dvc-dag:
	@echo "==> Affichage du graphe de dépendances DVC..."
	dvc dag

dvc-push:
	@echo "==> Push des artifacts vers le remote DVC..."
	dvc push

dvc-pull:
	@echo "==> Pull des artifacts depuis le remote DVC..."
	dvc pull

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	@echo "==> Lancement des tests unitaires..."
	pytest tests/ -v --tb=short

test-cov:
	@echo "==> Tests avec couverture de code..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ---------------------------------------------------------------------------
# Qualité de code
# ---------------------------------------------------------------------------
lint:
	@echo "==> Vérification du style (black + isort)..."
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	@echo "==> Formatage automatique..."
	black src/ tests/
	isort src/ tests/

# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------
clean:
	@echo "==> Nettoyage des fichiers temporaires..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/
	rm -f metrics.json
	@echo "==> Nettoyage terminé ✔"

# ---------------------------------------------------------------------------
# Aide
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "Akilang Whisper — Commandes disponibles :"
	@echo "------------------------------------------"
	@echo "  make train           Fine-tuning Whisper"
	@echo "  make evaluate        Évaluation + Quality Gate"
	@echo "  make export          Export ONNX + CTranslate2"
	@echo "  make pipeline        Pipeline complet"
	@echo "  make docker-train    Entraînement dans Docker"
	@echo "  make dvc-repro       Reproduction pipeline DVC"
	@echo "  make test            Tests unitaires"
	@echo "  make test-cov        Tests + couverture"
	@echo "  make lint            Vérification style"
	@echo "  make format          Formatage automatique"
	@echo "  make clean           Nettoyage temporaires"
	@echo ""
