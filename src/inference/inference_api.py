"""
src/inference/inference_api.py
API de transcription minimale utilisant Faster-Whisper (CTranslate2).

Ce script charge le modèle converti au format CTranslate2 et transcrit
un fichier audio passé en argument de ligne de commande.

Usage :
    python src/inference/inference_api.py <chemin_audio> [--model_dir <ct2_dir>] [--language fuv]

Exemple :
    python src/inference/inference_api.py audio.wav --model_dir outputs/export/ct2

Prérequis :
    pip install faster-whisper soundfile

Le modèle CTranslate2 doit être généré au préalable via :
    python src/export/export_model.py
"""

import argparse
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Analyse les arguments de ligne de commande.

    Returns:
        Namespace avec les champs : audio_path, model_dir, language, task,
        beam_size, device, compute_type
    """
    parser = argparse.ArgumentParser(
        description="Transcription Fulfuldé avec Faster-Whisper (CTranslate2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Chemin vers le fichier audio à transcrire
    parser.add_argument(
        "audio_path",
        type=str,
        help="Chemin vers le fichier audio (WAV, MP3, FLAC, etc.)",
    )

    # Répertoire contenant le modèle CTranslate2 converti
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("CT2_MODEL_DIR", "outputs/export/ct2"),
        help="Répertoire du modèle CTranslate2 (ou variable d'env CT2_MODEL_DIR)",
    )

    # Langue cible pour forcer la transcription en Fulfuldé
    parser.add_argument(
        "--language",
        type=str,
        default="fuv",
        help="Code de langue BCP-47 (fuv = Fulfuldé)",
    )

    # Tâche : transcription ou traduction
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Tâche Whisper",
    )

    # Largeur du beam search (qualité vs. vitesse)
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Largeur du beam search (plus grand = meilleur mais plus lent)",
    )

    # Device d'inférence
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device d'inférence",
    )

    # Type de compute (précision)
    parser.add_argument(
        "--compute_type",
        type=str,
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="Type de compute CTranslate2",
    )

    return parser.parse_args()


def transcribe(
    audio_path: str,
    model_dir: str,
    language: str = "fuv",
    task: str = "transcribe",
    beam_size: int = 5,
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """
    Transcrit un fichier audio avec Faster-Whisper.

    Args:
        audio_path   : chemin vers le fichier audio
        model_dir    : répertoire du modèle CTranslate2
        language     : code de langue (ex. "fuv" pour Fulfuldé)
        task         : "transcribe" ou "translate"
        beam_size    : largeur du beam search
        device       : "cpu", "cuda", ou "auto"
        compute_type : précision du calcul CTranslate2

    Returns:
        Transcription complète sous forme de chaîne de caractères
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper n'est pas installé. Exécutez : pip install faster-whisper")
        sys.exit(1)

    # Vérification que le modèle existe
    if not os.path.isdir(model_dir):
        logger.error(f"Répertoire modèle introuvable : {model_dir}")
        logger.error("Générez le modèle avec : python src/export/export_model.py")
        sys.exit(1)

    # Vérification que le fichier audio existe
    if not os.path.isfile(audio_path):
        logger.error(f"Fichier audio introuvable : {audio_path}")
        sys.exit(1)

    logger.info(f"Chargement du modèle depuis : {model_dir}")
    logger.info(f"Device={device}, compute_type={compute_type}")

    # Chargement du modèle Faster-Whisper
    # WhisperModel accepte un chemin local (model_size_or_path) ou un identifiant Hub
    model = WhisperModel(
        model_dir,
        device=device,
        compute_type=compute_type,
    )

    logger.info(f"Transcription de : {audio_path} (langue={language}, tâche={task})")

    # Transcription par segments (Faster-Whisper retourne un générateur)
    segments, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=True,       # Filtre de détection d'activité vocale (supprime les silences)
        vad_parameters={
            "min_silence_duration_ms": 500,  # Durée minimale d'un silence détecté
        },
    )

    logger.info(
        f"Langue détectée : {info.language} "
        f"(probabilité : {info.language_probability:.2%})"
    )

    # Assemblage des segments en une transcription complète
    transcription_parts = []
    for segment in segments:
        logger.debug(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text.strip()}")
        transcription_parts.append(segment.text.strip())

    full_transcription = " ".join(transcription_parts)
    return full_transcription


def main() -> None:
    """
    Point d'entrée CLI : parse les args, transcrit et affiche le résultat.
    """
    args = parse_args()

    result = transcribe(
        audio_path=args.audio_path,
        model_dir=args.model_dir,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    print("\n=== Transcription ===")
    print(result)
    print("====================\n")


if __name__ == "__main__":
    main()
