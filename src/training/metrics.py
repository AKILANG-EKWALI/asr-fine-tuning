"""
src/training/metrics.py
Calcul du WER (Word Error Rate) et du CER (Character Error Rate).

Ces métriques sont utilisées à deux endroits :
  1. Pendant l'entraînement (callback compute_metrics du Trainer HF)
  2. Lors de l'évaluation finale sur le split test

La normalisation du texte (minuscules, ponctuation) garantit une évaluation
équitable indépendante des variations de casse et de ponctuation.
"""

import re
import evaluate
from typing import NamedTuple


# =============================================================================
# CHARGEMENT DES MÉTRIQUES (une seule fois au niveau module pour performance)
# =============================================================================

# evaluate.load télécharge et met en cache les métriques HuggingFace
wer_metric = evaluate.load("wer")  # Word Error Rate (jiwer sous le capot)
cer_metric = evaluate.load("cer")  # Character Error Rate


# =============================================================================
# NORMALISATION DU TEXTE
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalise une chaîne de caractères pour l'évaluation WER/CER.

    Transformations appliquées :
      1. Minuscules          : "Bonjour" → "bonjour"
      2. Suppression ponctuation : "," "." "?" "!" → ""
         (regex \\W avec flag UNICODE pour couvrir les caractères Fulfuldé)
      3. Collapsage des espaces multiples : "a  b" → "a b"
      4. Strip des espaces de début/fin

    Note : les caractères Unicode non-ASCII (lettres Fulfuldé, diacritiques)
    sont CONSERVÉS par \\w en mode UNICODE.

    Args:
        text : texte brut (prédiction ou référence)

    Returns:
        Texte normalisé

    Exemples :
        >>> normalize_text("Bonjour, comment ça va ?")
        'bonjour comment ça va'
        >>> normalize_text("  mi  yahi  ")
        'mi yahi'
    """
    text = text.lower().strip()                              # Minuscules + strip
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)   # Supprime la ponctuation
    text = re.sub(r"\s+", " ", text)                         # Normalise les espaces
    return text.strip()


# =============================================================================
# CALCUL DES MÉTRIQUES (callback Trainer HF)
# =============================================================================

def compute_metrics(pred, tokenizer) -> dict:
    """
    Calcule WER et CER à partir des prédictions du Seq2SeqTrainer.

    Appelé automatiquement par le Trainer HuggingFace après chaque évaluation.
    Les scores sont multipliés par 100 pour être exprimés en %.

    Args:
        pred      : objet EvalPrediction (pred.predictions, pred.label_ids)
        tokenizer : WhisperTokenizer (pour décoder les token IDs)

    Returns:
        dict {"wer": float, "cer": float} en pourcentage

    Détail du pipeline :
      1. pred.label_ids contient -100 pour les tokens paddés — on les remplace
         par pad_token_id avant le décodage (sinon le tokenizer plante)
      2. On décode les séquences prédites et de référence en texte
      3. On normalise et on calcule les métriques
    """
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    # Remplace les -100 (tokens ignorés par la loss) par le pad_token_id
    # car batch_decode ne peut pas décoder -100
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Décodage des prédictions et labels : token IDs → texte
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Normalisation avant calcul des métriques (minuscules, sans ponctuation)
    pred_norm  = [normalize_text(s) for s in pred_str]
    label_norm = [normalize_text(s) for s in label_str]

    # Calcul WER : (substitutions + insertions + suppressions) / nb mots de référence
    wer = 100 * wer_metric.compute(predictions=pred_norm, references=label_norm)

    # Calcul CER : même principe mais au niveau des caractères
    cer = 100 * cer_metric.compute(predictions=pred_norm, references=label_norm)

    return {"wer": wer, "cer": cer}
