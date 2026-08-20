# =============================================================================
# tests/test_data.py
# Tests unitaires pour le module src/data/load_data.py
# Vérifie le filtre qualité sans accès fichier ni modèle HuggingFace.
# =============================================================================

import pytest
import numpy as np
from omegaconf import OmegaConf
from src.data.load_data import filter_example


# Configuration minimale partagée entre les tests
@pytest.fixture
def base_cfg():
    """Fixture : configuration Hydra minimale pour les tests data."""
    return OmegaConf.create({
        "data": {
            "duration_column":      "duration",
            "transcription_column": "sentence",
            "min_duration_sec":     0.5,
            "max_duration_sec":     30.0,
            "max_text_length":      200,
        }
    })


class TestFilterExample:
    """Tests de la fonction filter_example."""

    def test_rejects_audio_too_short(self, base_cfg):
        """Un audio de 0.1 s doit être rejeté (< 0.5 s)."""
        example = {"duration": 0.1, "sentence": "Bonjour"}
        assert not filter_example(example, base_cfg), \
            "Un audio trop court ne devrait pas passer le filtre"

    def test_rejects_audio_too_long(self, base_cfg):
        """Un audio de 31 s doit être rejeté (> 30 s)."""
        example = {"duration": 31.0, "sentence": "Bonjour"}
        assert not filter_example(example, base_cfg), \
            "Un audio trop long ne devrait pas passer le filtre"

    def test_accepts_valid_example(self, base_cfg):
        """Un exemple valide (5 s, texte court) doit passer."""
        example = {"duration": 5.0, "sentence": "Bonjour mi yahi"}
        assert filter_example(example, base_cfg), \
            "Un exemple valide devrait passer le filtre"

    def test_rejects_empty_sentence(self, base_cfg):
        """Une transcription vide doit être rejetée."""
        example = {"duration": 3.0, "sentence": ""}
        assert not filter_example(example, base_cfg)

    def test_rejects_whitespace_only_sentence(self, base_cfg):
        """Une transcription d'espaces seulement doit être rejetée."""
        example = {"duration": 3.0, "sentence": "   "}
        assert not filter_example(example, base_cfg)

    def test_rejects_none_sentence(self, base_cfg):
        """Une transcription None doit être rejetée."""
        example = {"duration": 3.0, "sentence": None}
        assert not filter_example(example, base_cfg)

    def test_rejects_text_too_long(self, base_cfg):
        """Un texte de 250 mots doit être rejeté (> max_text_length=200)."""
        long_text = " ".join(["mot"] * 250)
        example = {"duration": 5.0, "sentence": long_text}
        assert not filter_example(example, base_cfg)

    def test_accepts_example_without_duration_column(self, base_cfg):
        """Si la colonne duration est absente, l'exemple est accepté si le texte est valide."""
        example = {"sentence": "Bonjour"}  # Pas de clé "duration"
        assert filter_example(example, base_cfg), \
            "Un exemple sans durée mais avec texte valide devrait passer"

    def test_accepts_boundary_duration_min(self, base_cfg):
        """Une durée exactement à la limite basse (0.5 s) doit être acceptée."""
        example = {"duration": 0.5, "sentence": "Bonjour"}
        assert filter_example(example, base_cfg)

    def test_accepts_boundary_duration_max(self, base_cfg):
        """Une durée exactement à la limite haute (30 s) doit être acceptée."""
        example = {"duration": 30.0, "sentence": "Bonjour"}
        assert filter_example(example, base_cfg)
