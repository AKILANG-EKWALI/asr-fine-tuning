# Akilang Whisper 🎙️

> **Pipeline MLOps complet pour le fine-tuning de Whisper sur le Fulfuldé**  
> Variante linguistique du Grand Nord Cameroun · LoRA/PEFT · MLflow · DVC · Docker · ONNX · Faster-Whisper

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis système](#prérequis-système)
3. [Installation](#installation)
4. [Préparer les données](#préparer-les-données)
5. [Lancer l'entraînement](#lancer-lentraînement)
6. [Évaluation & Quality Gate](#évaluation--quality-gate)
7. [Exporter le modèle](#exporter-le-modèle)
8. [Inférence](#inférence)
9. [Déploiement Docker](#déploiement-docker)
10. [Pipeline DVC (reproductibilité)](#pipeline-dvc-reproductibilité)
11. [Suivi MLflow](#suivi-mlflow)
12. [Configuration Hydra (zero hardcoding)](#configuration-hydra-zero-hardcoding)
13. [Tests unitaires](#tests-unitaires)
14. [Augmentation téléphonique](#augmentation-téléphonique)
15. [Quality Gate](#quality-gate)
16. [Structure du projet](#structure-du-projet)
17. [Dépannage](#dépannage)
18. [Contribuer](#contribuer)
19. [Licence](#licence)

---

## Vue d'ensemble

Akilang Whisper est un pipeline de fine-tuning production-ready pour adapter [OpenAI Whisper](https://github.com/openai/whisper) au **Fulfuldé** (Fula, code BCP-47 : `fuv`), une langue parlée par plusieurs millions de personnes au Cameroun, au Nigeria et en Afrique de l'Ouest.

### Fonctionnalités clés

| Fonctionnalité | Détail |
|----------------|--------|
| **Modèle de base** | `openai/whisper-base` (extensible à `small`, `medium`, `large`) |
| **Fine-tuning efficace** | LoRA via PEFT — < 1 % des paramètres entraînés, 4× moins de VRAM |
| **Augmentation téléphonique** | Filtre 300–3400 Hz + bruit SNR 5–15 dB + simulation G.711 |
| **Suivi d'expériences** | MLflow — métriques, paramètres, modèles versionés |
| **Reproductibilité** | DVC — pipeline versioné, données et modèles trackés |
| **Quality Gate** | Seuils WER/CER configurables avant autorisation d'export |
| **Export production** | ONNX (optimum) + CTranslate2 / Faster-Whisper (quantization INT8) |
| **Conteneurisation** | Docker avec support GPU (NVIDIA) + image slim pour inférence |
| **Configuration** | Hydra — zéro hardcoding, overrides CLI, configs hiérarchiques |

---

## Prérequis système

### Pour l'entraînement (recommandé)

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **GPU** | NVIDIA 8 Go VRAM (RTX 3060) | NVIDIA 24 Go (RTX 3090 / A10G) |
| **RAM** | 16 Go | 32 Go |
| **Stockage** | 50 Go libres | 100 Go SSD |
| **OS** | Ubuntu 20.04+ / macOS 13+ | Ubuntu 22.04 |
| **Python** | 3.10 | 3.10 |
| **CUDA** | 11.8+ | 12.1 |

### Pour l'inférence uniquement (CPU possible)

- Python 3.10+, 4 Go RAM, pas de GPU requis

### Logiciels requis

```bash
# Vérifier Python
python --version          # Python 3.10.x

# Vérifier CUDA (pour GPU)
nvidia-smi                # Affiche la version CUDA
nvcc --version

# Vérifier ffmpeg (requis pour le décodage audio)
ffmpeg -version

# Docker (optionnel)
docker --version          # Docker 24.0+
docker compose version    # Docker Compose 2.0+
```

> **macOS (Apple Silicon)** : L'entraînement fonctionne sur MPS (Metal Performance Shaders).  
> Ajoutez `training.fp16=false` aux overrides Hydra car fp16 n'est pas supporté sur MPS.

---

## Installation

### Option A — Installation locale (recommandée pour le développement)

```bash
# 1. Cloner le repository
git clone https://github.com/votre-org/akilang-whisper.git
cd akilang-whisper

# 2. Créer un environnement virtuel Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

# 3. Mettre à jour pip
pip install --upgrade pip

# 4. Installer les dépendances d'entraînement
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('GPU disponible:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

> **⚠️ Windows** : Certaines dépendances (`audiomentations`, `librosa`) nécessitent  
> Microsoft Build Tools. Préférez WSL2 ou Docker.

### Option B — Environnement Conda

```bash
conda create -n akilang python=3.10 -y
conda activate akilang
conda install -c pytorch -c nvidia pytorch=2.1.0 torchvision torchaudio cudatoolkit=12.1 -y
pip install -r requirements.txt
```

### Option C — Docker (aucune installation locale requise)

Voir la section [Déploiement Docker](#déploiement-docker).

### Vérification complète de l'installation

```bash
# Lance les tests unitaires pour vérifier que tout fonctionne
make test
```

---

## Préparer les données

### Structure attendue

Placez vos données dans le dossier `data/clean/` avec cette structure **exacte** :

```
data/clean/
├── clips/              ← Fichiers audio (WAV 16 kHz ou MP3)
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
├── train.tsv           ← Split entraînement
├── dev.tsv             ← Split validation
└── test.tsv            ← Split test final
```

### Format des fichiers TSV

Chaque TSV est **séparé par des tabulations** (`\t`) avec au minimum ces colonnes :

```tsv
path	sentence	duration
sample_001.wav	Mi yahi e nder suudu	3.25
sample_002.wav	Ɓe ngertii ko fii	2.10
```

| Colonne | Obligatoire | Description |
|---------|-------------|-------------|
| `path` | ✅ | Chemin relatif au dossier `clips/` |
| `sentence` | ✅ | Transcription en Fulfuldé |
| `duration` | ❌ (optionnel) | Durée en secondes (utilisée pour le filtrage) |

> **Common Voice Fulfuldé** : Si vous utilisez le dataset Mozilla Common Voice,  
> les TSV sont déjà dans ce format. Téléchargez depuis [commonvoice.mozilla.org](https://commonvoice.mozilla.org/fr/datasets).

### Préparer les bruits (optionnel, pour l'augmentation)

```bash
# Placez des fichiers audio de bruit ambiant dans data/noise/
# Sources recommandées :
# - MUSAN : https://www.openslr.org/17/
# - ESC-50 : https://github.com/karolpiczak/ESC-50

ls data/noise/
# noise_street.wav  noise_cafe.wav  noise_wind.wav ...
```

Si `data/noise/` est vide, seul le bruit gaussien (AddGaussianSNR) sera utilisé — le pipeline fonctionne quand même.

### Validation des données

```bash
# Vérifier le nombre d'exemples dans chaque split
wc -l data/clean/train.tsv data/clean/dev.tsv data/clean/test.tsv

# Vérifier que les fichiers audio existent
head -5 data/clean/train.tsv | awk -F'\t' '{print "data/clean/clips/" $1}' | xargs ls -la

# Vérifier le format (doit afficher les colonnes)
head -2 data/clean/train.tsv
```

---

## Lancer l'entraînement

### Démarrage rapide (1 commande)

```bash
make train
```

### Options d'entraînement

```bash
# Entraînement standard (config par défaut : 5 epochs, batch=8, LoRA r=8)
make train

# Avec overrides Hydra (modifier les hyperparamètres à la volée)
make train HYDRA_OVERRIDES="training.num_train_epochs=10"
make train HYDRA_OVERRIDES="training.learning_rate=5e-5 training.per_device_train_batch_size=4"

# Désactiver l'augmentation téléphonique
make train HYDRA_OVERRIDES="data.augmentation.enabled=false"

# Changer de modèle (plus grand = meilleur WER mais plus de VRAM)
make train HYDRA_OVERRIDES="model.name=openai/whisper-small"
make train HYDRA_OVERRIDES="model.name=openai/whisper-medium"

# Fine-tuning complet sans LoRA (nécessite ~4× plus de VRAM)
make train HYDRA_OVERRIDES="model.use_peft=false"

# Via Docker avec GPU
make docker-train
```

### Ce qui se passe pendant l'entraînement

```
outputs/akilang_whisper_fuv/
└── 2024-01-15_10-30-00/
    └── checkpoints/
        ├── checkpoint-500/          ← Sauvegardé tous les 500 pas
        ├── checkpoint-1000/
        ├── checkpoint-best/         ← Meilleur checkpoint (WER le plus bas)
        └── final/                   ← Modèle final après toutes les epochs
```

Les métriques WER et CER sont affichées en temps réel dans la console et loguées dans MLflow.

### Reprendre un entraînement interrompu

```bash
# DVC mémorise l'état du pipeline — relancer dvc repro ne réentraîne pas si rien n'a changé
dvc repro

# Pour forcer la réexécution d'une étape spécifique
dvc repro -f train
```

---

## Évaluation & Quality Gate

```bash
# Évaluation complète (calcul WER/CER + vérification des seuils)
make evaluate
```

Cette commande exécute deux étapes :

**Étape 1 — `evaluate.py`** : calcule WER et CER sur le split test et produit `metrics.json` :
```json
{
    "test_wer": 18.42,
    "test_cer": 22.15,
    "test_loss": 0.38,
    "test_runtime": 124.3
}
```

**Étape 2 — `quality_gate.py`** : compare les métriques aux seuils configurés :
```
========================================
QUALITY GATE — Résultats
========================================
  WER test : 18.42%  (seuil max : 25.0%)
  CER test : 22.15%  (seuil max : 30.0%)
========================================
✔  Quality Gate PASSÉ — Export autorisé
```

Si le gate **échoue** → le script retourne un code d'erreur non-nul, ce qui bloque le pipeline DVC et la CI/CD.

---

## Exporter le modèle

```bash
# Export après passage du quality gate
make export
```

Trois sorties sont produites :

| Format | Dossier | Usage |
|--------|---------|-------|
| HuggingFace (fusionné) | `outputs/.../checkpoints/merged/` | Référence, fine-tuning ultérieur |
| ONNX (opset 14) | `outputs/.../export/onnx/` | Inférence via onnxruntime |
| CTranslate2 (INT8) | `outputs/.../export/ct2/` | Inférence ultra-rapide via Faster-Whisper |

### Options d'export

```bash
# Changer la quantification (float16 pour GPU, int8 pour CPU)
make export HYDRA_OVERRIDES="mlops.export.quantization=float16"
make export HYDRA_OVERRIDES="mlops.export.quantization=int8"
make export HYDRA_OVERRIDES="mlops.export.quantization=int8_float16"

# Exporter vers un dossier personnalisé
make export HYDRA_OVERRIDES="mlops.export.ct2_dir=/models/akilang_v1"
```

---

## Inférence

### Inférence CLI (après export)

```bash
# Transcrire un fichier audio
python src/inference/inference_api.py mon_audio.wav

# Spécifier le répertoire du modèle CTranslate2
python src/inference/inference_api.py audio.wav --model_dir outputs/export/ct2

# Options avancées
python src/inference/inference_api.py audio.wav \
    --model_dir outputs/export/ct2 \
    --language fuv \
    --beam_size 10 \
    --device cpu \
    --compute_type int8
```

### Options de l'API d'inférence

| Argument | Défaut | Description |
|----------|--------|-------------|
| `audio_path` | (requis) | Chemin vers le fichier audio (WAV, MP3, FLAC) |
| `--model_dir` | `outputs/export/ct2` | Répertoire du modèle CTranslate2 |
| `--language` | `fuv` | Code de langue BCP-47 |
| `--beam_size` | `5` | Largeur beam search (qualité vs vitesse) |
| `--device` | `cpu` | `cpu`, `cuda`, ou `auto` |
| `--compute_type` | `int8` | Précision : `int8`, `float16`, `int8_float16` |

### Inférence Docker

```bash
# Transcrire un fichier audio via Docker
docker compose run --rm -v /chemin/vers/audio:/audio inference /audio/test.wav

# Avec variable d'env pour le modèle
docker compose run --rm \
    -e CT2_MODEL_DIR=/app/outputs/export/ct2 \
    -v ./outputs:/app/outputs \
    -v /chemin/vers/audio:/audio \
    inference /audio/test.wav
```

---

## Déploiement Docker

### Prérequis Docker

```bash
# Installer Docker
# → https://docs.docker.com/engine/install/

# Installer nvidia-container-toolkit (pour GPU dans Docker)
# → https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Vérifier que le GPU est accessible dans Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Build des images

```bash
# Build de l'image d'entraînement (GPU, ~6 Go)
docker compose build train

# Build de l'image d'inférence (CPU, ~1.5 Go)
docker compose build inference

# Build toutes les images
make docker-build
```

### Services disponibles

```bash
# Entraînement avec GPU
docker compose run --rm train

# Évaluation + quality gate
docker compose run --rm evaluate

# Export ONNX + CTranslate2
docker compose run --rm export

# Inférence CPU
docker compose run --rm inference mon_audio.wav

# MLflow UI (http://localhost:5000)
docker compose --profile monitoring up mlflow-ui
```

### Override Hydra dans Docker

```bash
# Passer des overrides Hydra à l'entraînement Docker
docker compose run --rm train training.num_train_epochs=3 data.augmentation.enabled=false
```

---

## Pipeline DVC (reproductibilité)

DVC (Data Version Control) permet de :
- Versionner les données volumineuses sans les mettre dans Git
- Reproduire exactement le même pipeline sur n'importe quelle machine
- Partager les données et modèles via un stockage distant (S3, GCS, etc.)

### Initialisation DVC (première fois)

```bash
# Initialiser DVC dans le projet
git init
dvc init
git add .dvc .gitignore
git commit -m "Initialisation DVC"

# (Optionnel) Configurer un remote DVC pour partager les données
dvc remote add -d myremote s3://mon-bucket/akilang-whisper
# ou
dvc remote add -d myremote gdrive://mon-folder-id

# Versionner les données locales
dvc add data/clean
git add data/clean.dvc .gitignore
git commit -m "Ajout dataset Common Voice Fulfuldé"
dvc push
```

### Utilisation quotidienne

```bash
# Reproduire le pipeline complet (skip les étapes non modifiées)
dvc repro

# Voir quelles étapes seront réexécutées
dvc status

# Afficher le graphe de dépendances
dvc dag
# Output :
#   +-------+
#   | train |
#   +-------+
#       |
#       v
#  +----------+
#  | evaluate |
#  +----------+
#       |
#       v
# +--------------+
# | quality_gate |
# +--------------+
#       |
#       v
#   +--------+
#   | export |
#   +--------+

# Voir les métriques de toutes les expériences
dvc metrics show

# Comparer deux runs
dvc metrics diff HEAD~1

# Partager les artifacts avec l'équipe
dvc push
dvc pull   # Sur une autre machine
```

---

## Suivi MLflow

Toutes les métriques, paramètres et modèles sont automatiquement loggés dans MLflow.

### Accéder à l'interface MLflow

```bash
# Option 1 : Lancer l'UI localement
mlflow ui --backend-store-uri ./mlflow --port 5000
# → Ouvrir http://localhost:5000

# Option 2 : Via Docker
docker compose --profile monitoring up mlflow-ui
# → Ouvrir http://localhost:5000
```

### Ce qui est loggé automatiquement

- **Paramètres** : tous les hyperparamètres Hydra (modèle, training, data, mlops)
- **Métriques** : WER et CER à chaque évaluation, `best_wer` en fin de run
- **Artefacts** : le modèle final si `mlops.mlflow.log_model=true`

### Configurer un serveur MLflow distant

```bash
# Dans configs/mlops/default.yaml
# Remplacer :
#   tracking_uri: "file:///mlflow"
# Par :
#   tracking_uri: "http://mlflow-server:5000"

# Ou via override Hydra
make train HYDRA_OVERRIDES="mlops.mlflow.tracking_uri=http://mlflow-server:5000"
```

---

## Configuration Hydra (zero hardcoding)

Toute la configuration est dans `configs/`. Aucune valeur n'est codée en dur dans le code Python.

### Structure des configurations

```
configs/
├── config.yaml                   ← Point d'entrée (charge les 4 sous-configs)
├── data/commonvoice_fuv.yaml     ← Chemins, filtres, augmentation
├── model/whisper_base_lora.yaml  ← Modèle, LoRA (r, alpha, modules)
├── training/default.yaml         ← Hyperparamètres d'entraînement
└── mlops/default.yaml            ← MLflow, quality gate, export
```

### Modifier la configuration

**Méthode 1 — Override CLI (recommandé, sans modifier les fichiers)**
```bash
python src/training/train.py training.learning_rate=5e-5 training.num_train_epochs=10
```

**Méthode 2 — Modifier le fichier YAML**
```yaml
# configs/training/default.yaml
num_train_epochs: 10          # Modifier directement
learning_rate: 5e-5
```

**Méthode 3 — Créer une nouvelle config**
```bash
# Créer une config pour whisper-small
cp configs/model/whisper_base_lora.yaml configs/model/whisper_small_lora.yaml
# Modifier whisper_small_lora.yaml : name: "openai/whisper-small"

# Utiliser cette config
python src/training/train.py model=whisper_small_lora
```

### Référence des paramètres clés

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `model.name` | `openai/whisper-base` | Identifiant HuggingFace du modèle |
| `model.lora.r` | `8` | Rang LoRA (↑ = plus de capacité, plus de VRAM) |
| `training.num_train_epochs` | `5` | Nombre d'époques |
| `training.learning_rate` | `1e-4` | Taux d'apprentissage |
| `training.per_device_train_batch_size` | `8` | Batch par GPU |
| `training.fp16` | `true` | Mixed precision (désactiver sur CPU / MPS) |
| `data.augmentation.enabled` | `true` | Augmentation téléphonique |
| `data.augmentation.prob` | `0.7` | Probabilité d'augmentation par exemple |
| `mlops.quality_gate.max_wer` | `25.0` | Seuil WER maximal (%) |
| `mlops.export.quantization` | `int8_float16` | Quantification CTranslate2 |

---

## Tests unitaires

```bash
# Lancer tous les tests
make test

# Mode verbeux avec détail des tests
pytest tests/ -v

# Tests avec rapport de couverture
make test-cov
# → Rapport HTML dans htmlcov/index.html

# Lancer uniquement un fichier de test
pytest tests/test_data.py -v
pytest tests/test_augment.py -v
pytest tests/test_metrics.py -v

# Lancer un test spécifique
pytest tests/test_data.py::TestFilterExample::test_rejects_audio_too_short -v
```

### Tests disponibles

| Fichier | Tests | Ce qui est vérifié |
|---------|-------|-------------------|
| `test_data.py` | 10 | Filtre qualité (durée, texte, cas limites) |
| `test_metrics.py` | 9 | Normalisation texte (ponctuation, Unicode, espaces) |
| `test_augment.py` | 5 | Shape signal, float32, désactivation, longueurs variées |
| `test_export.py` | 5 | Chemins export, logique quality gate, structure JSON |

---

## Augmentation téléphonique

Le pipeline simule les conditions d'un appel téléphonique (qualité 8 kHz) pour rendre le modèle robuste aux données vocales de faible qualité.

### Pipeline d'augmentation

```
Signal 16 kHz original
       │
       ▼
┌─────────────────────────────────┐
│  Filtre passe-bande 300–3400 Hz │  ← Bande passante téléphonique
│  (probabilité : 80%)            │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Bruit gaussien SNR 5–15 dB     │  ← Bruit de ligne
│  (probabilité : 80%)            │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  16 kHz → 8 kHz → 16 kHz       │  ← Simulation codec G.711
│  (toujours si enabled=true)     │
└─────────────────────────────────┘
       │
       ▼
Signal augmenté 16 kHz (même shape)
```

### Configuration

```yaml
# configs/data/commonvoice_fuv.yaml
augmentation:
  enabled: true     # Désactiver : make train HYDRA_OVERRIDES="data.augmentation.enabled=false"
  prob: 0.7         # 70% des exemples d'entraînement sont augmentés
  bandpass:
    min_center_freq: 300
    max_center_freq: 3400
    p: 0.8
  noise:
    min_snr_db: 5   # Bruit fort (difficile)
    max_snr_db: 15  # Bruit léger (facile)
    p: 0.8
```

---

## Quality Gate

Le quality gate bloque automatiquement l'export si les métriques sont insuffisantes.

### Seuils par défaut

| Métrique | Seuil | Signification |
|----------|-------|---------------|
| **WER** ≤ 25% | Bloquant | 1 mot sur 4 mal transcrit au maximum |
| **CER** ≤ 30% | Bloquant | 30% de caractères incorrects au maximum |

### Ajuster les seuils

```bash
# Seuils plus stricts
make evaluate HYDRA_OVERRIDES="mlops.quality_gate.max_wer=20.0 mlops.quality_gate.max_cer=25.0"

# Via le fichier de config
# → Modifier configs/mlops/default.yaml → quality_gate
```

### Comportement en CI/CD

```yaml
# Exemple GitHub Actions
- name: Quality Gate
  run: make evaluate
  # Retourne exit code 1 si WER > 25% ou CER > 30%
  # → La CI échoue automatiquement
```

---

## Structure du projet

```
akilang-whisper/
├── configs/                          # Configs Hydra (zero hardcoding)
│   ├── config.yaml                   # Point d'entrée — charge les 4 sous-configs
│   ├── data/commonvoice_fuv.yaml     # Chemins, filtres, augmentation
│   ├── model/whisper_base_lora.yaml  # Modèle + LoRA (r, alpha, modules)
│   ├── training/default.yaml         # Hyperparamètres entraînement
│   └── mlops/default.yaml            # MLflow, quality gate, export
│
├── src/                              # Code source Python
│   ├── data/
│   │   ├── load_data.py              # Lecture TSV, filtrage, préparation Whisper
│   │   ├── augment.py                # Augmentation téléphonique (bandpass + bruit + G.711)
│   │   └── collator.py              # DataCollator padding pour Seq2Seq
│   ├── model/
│   │   ├── build_model.py           # Chargement Whisper + wrapping LoRA
│   │   └── lora_utils.py            # Détection auto des couches linéaires
│   ├── training/
│   │   ├── train.py                 # Script principal d'entraînement (entrypoint)
│   │   └── metrics.py               # WER/CER + normalisation texte
│   ├── eval/
│   │   ├── evaluate.py              # Évaluation split test → metrics.json
│   │   └── quality_gate.py          # Vérification seuils WER/CER
│   ├── export/
│   │   └── export_model.py          # Fusion LoRA + export ONNX + CTranslate2
│   └── inference/
│       └── inference_api.py         # CLI de transcription Faster-Whisper
│
├── scripts/                          # Scripts shell
│   ├── train.sh                      # Lance train.py (avec set -euo pipefail)
│   ├── evaluate.sh                   # Lance evaluate.py + quality_gate.py
│   └── export.sh                     # Lance export_model.py
│
├── tests/                            # Tests unitaires pytest
│   ├── test_data.py                  # Tests filtrage qualité (10 cas)
│   ├── test_metrics.py               # Tests normalisation texte (9 cas)
│   ├── test_augment.py               # Tests augmentation (5 cas)
│   └── test_export.py                # Tests logique export/gate (5 cas)
│
├── data/
│   ├── clean/                        # Dataset local (géré par DVC)
│   │   ├── clips/                    # Fichiers audio WAV/MP3
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   └── noise/                        # Bruits ambiants (optionnel)
│
├── Dockerfile.train                  # Image GPU (pytorch:2.1.0-cuda12.1)
├── Dockerfile.inference              # Image slim CPU (python:3.10-slim)
├── docker-compose.yml                # 5 services : train, evaluate, export, inference, mlflow-ui
├── dvc.yaml                          # Pipeline DVC : train → evaluate → quality_gate → export
├── Makefile                          # 15 cibles : train, evaluate, export, test, lint...
├── requirements.txt                  # Dépendances entraînement (épinglées)
├── requirements_inference.txt        # Dépendances inférence (sous-ensemble minimal)
├── pyproject.toml                    # Config Black, isort, pytest, coverage
├── .gitignore                        # Exclut outputs/, mlflow/, data/clean/clips/
├── README.md                         # Ce fichier
└── LICENSE                           # Apache 2.0
```

---

## Dépannage

### `CUDA out of memory`

```bash
# Réduire le batch size
make train HYDRA_OVERRIDES="training.per_device_train_batch_size=4 training.gradient_accumulation_steps=4"

# Activer le gradient checkpointing (déjà activé par défaut)
make train HYDRA_OVERRIDES="training.gradient_checkpointing=true"

# Utiliser un modèle plus petit
make train HYDRA_OVERRIDES="model.name=openai/whisper-tiny"
```

### `ModuleNotFoundError: No module named 'src'`

```bash
# Lancer les scripts depuis la racine du projet
cd akilang-whisper
python src/training/train.py    # ✅ Correct
# python train.py               # ❌ Mauvais répertoire

# Ou ajouter le projet au PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### `FileNotFoundError: data/clean/train.tsv`

```bash
# Vérifier que les données sont bien placées
ls data/clean/
# → clips/  dev.tsv  test.tsv  train.tsv

# Le chemin dans la config est relatif au répertoire d'exécution
# Toujours lancer depuis la racine du projet
```

### `ffmpeg: command not found`

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

### Quality Gate échoue avec WER > 25%

```bash
# Vérifier que le modèle est correctement sauvegardé
ls outputs/*/checkpoints/final/

# Relancer l'évaluation avec un seuil plus souple (pour déboguer)
make evaluate HYDRA_OVERRIDES="mlops.quality_gate.max_wer=50.0 mlops.quality_gate.max_cer=60.0"

# Augmenter le nombre d'epochs d'entraînement
make train HYDRA_OVERRIDES="training.num_train_epochs=10"
```

### `optimum-cli: command not found` (lors de l'export)

```bash
# Vérifier l'installation
pip show optimum

# Réinstaller
pip install "optimum[onnxruntime]"

# Vérifier la commande
optimum-cli --help
```

---

## Contribuer

1. Forkez le repository
2. Créez une branche : `git checkout -b feature/ma-fonctionnalite`
3. Vérifiez le style : `make lint`
4. Lancez les tests : `make test`
5. Committez : `git commit -m "feat: ajout de ma fonctionnalité"`
6. Ouvrez une Pull Request

### Standards de code

```bash
# Formater le code avant de committer
make format       # Black + isort
make lint         # Vérification (sans modification)
make test         # Tests unitaires obligatoires
```

---

## Licence

Ce projet est distribué sous licence **Apache 2.0** — voir [LICENSE](LICENSE).

---

*Akilang — Valoriser les langues d'Afrique par la technologie vocale* 🌍
