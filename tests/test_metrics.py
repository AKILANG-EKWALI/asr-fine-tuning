# =============================================================================
# tests/test_metrics.py
# Tests unitaires pour src/training/metrics.py
# Vérifie la normalisation du texte et le calcul WER/CER.
# =============================================================================

import pytest
from src.training.metrics import normalize_text


class TestNormalizeText:
    """Tests de la fonction normalize_text."""

    def test_removes_punctuation(self):
        """La ponctuation française doit être supprimée."""
        result = normalize_text("Bonjour, comment ça va ?")
        assert result == "bonjour comment ça va", \
            f"Résultat inattendu : '{result}'"

    def test_lowercases(self):
        """Le texte doit être converti en minuscules."""
        assert normalize_text("BONJOUR") == "bonjour"

    def test_collapses_whitespace(self):
        """Les espaces multiples doivent être collapsés en un seul."""
        assert normalize_text("mi   yahi") == "mi yahi"

    def test_strips_leading_trailing_spaces(self):
        """Les espaces en début et fin doivent être supprimés."""
        assert normalize_text("  mi yahi  ") == "mi yahi"

    def test_preserves_unicode_letters(self):
        """Les lettres Unicode Fulfuldé (caractères accentués) doivent être conservées."""
        text = "ɓe ngertii"
        result = normalize_text(text)
        # Les lettres Unicode doivent être préservées (regex \\w UNICODE)
        assert "ɓe" in result, f"Caractère Unicode perdu dans : '{result}'"

    def test_empty_string(self):
        """Une chaîne vide doit retourner une chaîne vide."""
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        """Une chaîne d'espaces seulement doit retourner une chaîne vide."""
        assert normalize_text("   ") == ""

    def test_removes_exclamation_and_question(self):
        """Les points d'exclamation et d'interrogation doivent être supprimés."""
        result = normalize_text("Mbele! Mbele?")
        assert "!" not in result
        assert "?" not in result

    def test_already_normalized(self):
        """Un texte déjà normalisé ne doit pas être modifié."""
        text = "mi yahi"
        assert normalize_text(text) == text
