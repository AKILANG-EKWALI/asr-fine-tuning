# =============================================================================
# tests/test_augment.py
# Tests unitaires pour src/data/augment.py
# Vérifie que l'augmentation téléphonique ne modifie pas la forme du signal.
# =============================================================================

import pytest
import numpy as np
from omegaconf import OmegaConf
from src.data.augment import TelephoneAugmenter


@pytest.fixture
def augment_cfg():
    """Fixture : configuration d'augmentation avec toutes les transformations actives."""
    return OmegaConf.create({
        "data": {
            "augmentation": {
                "enabled": True,
                "prob":    1.0,   # Force l'application pour les tests déterministes
                "bandpass": {
                    "min_center_freq": 300,
                    "max_center_freq": 3400,
                    "p": 1.0,
                },
                "noise": {
                    "noise_dir":  "./data/noise",
                    "min_snr_db": 5,
                    "max_snr_db": 15,
                    "p": 1.0,
                },
                "g711": {
                    "enabled": True,
                },
            }
        }
    })


@pytest.fixture
def disabled_cfg():
    """Fixture : configuration d'augmentation désactivée."""
    return OmegaConf.create({
        "data": {
            "augmentation": {
                "enabled": False,
                "prob":    0.0,
                "bandpass": {"min_center_freq": 300, "max_center_freq": 3400, "p": 1.0},
                "noise":    {"noise_dir": "./data/noise", "min_snr_db": 5, "max_snr_db": 15, "p": 1.0},
                "g711":     {"enabled": False},
            }
        }
    })


class TestTelephoneAugmenter:
    """Tests de la classe TelephoneAugmenter."""

    def test_output_shape_matches_input(self, augment_cfg):
        """La forme du signal de sortie doit être identique à celle de l'entrée."""
        augmenter = TelephoneAugmenter(augment_cfg)
        audio = np.random.randn(16000).astype(np.float32)  # 1 seconde à 16 kHz
        output = augmenter(audio, sample_rate=16000)
        assert output.shape == audio.shape, \
            f"Shape entrée={audio.shape}, shape sortie={output.shape}"

    def test_output_is_float32(self, augment_cfg):
        """Le signal de sortie doit être en float32."""
        augmenter = TelephoneAugmenter(augment_cfg)
        audio = np.random.randn(16000).astype(np.float32)
        output = augmenter(audio, sample_rate=16000)
        assert output.dtype == np.float32, f"dtype attendu float32, obtenu {output.dtype}"

    def test_disabled_returns_original(self, disabled_cfg):
        """Quand l'augmentation est désactivée, le signal doit être retourné inchangé."""
        augmenter = TelephoneAugmenter(disabled_cfg)
        audio = np.random.randn(8000).astype(np.float32)
        output = augmenter(audio, sample_rate=16000)
        np.testing.assert_array_equal(output, audio, err_msg="Le signal ne devrait pas être modifié")

    def test_different_lengths(self, augment_cfg):
        """L'augmentation doit fonctionner pour différentes longueurs de signal."""
        augmenter = TelephoneAugmenter(augment_cfg)
        for n_samples in [8000, 16000, 48000]:  # 0.5s, 1s, 3s à 16 kHz
            audio = np.random.randn(n_samples).astype(np.float32)
            output = augmenter(audio, sample_rate=16000)
            assert output.shape == audio.shape, \
                f"Échec pour n_samples={n_samples} : shape={output.shape}"

    def test_output_not_all_zeros(self, augment_cfg):
        """Le signal augmenté ne doit pas être tout à zéro (signal corrompu)."""
        augmenter = TelephoneAugmenter(augment_cfg)
        audio = np.random.randn(16000).astype(np.float32)
        output = augmenter(audio, sample_rate=16000)
        assert not np.all(output == 0), "Le signal augmenté est entièrement nul"
