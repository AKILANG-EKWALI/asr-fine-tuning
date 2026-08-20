"""
src/data/augment.py
Augmentation téléphonique du signal audio.

Pipeline appliqué (probabiliste) :
  1. Filtre passe-bande 300–3400 Hz  → simule la bande passante téléphonique
  2. Bruit additif gaussien (SNR 5–15 dB) → simule les bruits de ligne
  3. Rééchantillonnage 16 kHz → 8 kHz → 16 kHz (codec G.711) → dégrade la qualité

L'augmentation est activée par cfg.data.augmentation.enabled et appliquée
avec probabilité cfg.data.augmentation.prob par exemple.
"""

import random
import numpy as np
import torch
import torchaudio
try:
    import audiomentations as A
except ImportError:
    A = None
from omegaconf import DictConfig


class TelephoneAugmenter:
    """
    Classe d'augmentation téléphonique réutilisable.

    Exemple d'utilisation :
        augmenter = TelephoneAugmenter(cfg)
        augmented = augmenter(audio_array, sample_rate=16000)
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialise les transformations à partir de la config Hydra.

        Args:
            cfg : configuration Hydra complète (on lit cfg.data.augmentation.*)
        """
        self.cfg = cfg
        self.enabled = cfg.data.augmentation.enabled   # Killswitch global
        self.prob    = cfg.data.augmentation.prob       # Probabilité par exemple

        if self.enabled:
            if A is None:
                raise ImportError(
                    "Le module 'audiomentations' n'est pas disponible dans votre environnement. "
                    "Veuillez l'installer (ex: pip install audiomentations) ou désactivez "
                    "les augmentations dans la config (ex: data.augmentation.enabled=false)."
                )
            # --- Filtre passe-bande (audiomentations) ---
            self.bandpass = A.BandPassFilter(
                min_center_frequency=cfg.data.augmentation.bandpass.min_center_freq,
                max_center_frequency=cfg.data.augmentation.bandpass.max_center_freq,
                min_bandwidth_fraction=0.5,
                max_bandwidth_fraction=0.8,
                p=cfg.data.augmentation.bandpass.p,
            )

            # --- Bruit additif gaussien (audiomentations) ---
            self.noise_aug = A.AddGaussianSNR(
                min_snr_db=cfg.data.augmentation.noise.min_snr_db,
                max_snr_db=cfg.data.augmentation.noise.max_snr_db,
                p=cfg.data.augmentation.noise.p,
            )
        else:
            self.bandpass = None
            self.noise_aug = None

    def __call__(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applique la chaîne d'augmentation téléphonique sur un signal audio.

        Args:
            audio_array : signal numpy float32 de shape (N,) ou (1, N)
            sample_rate : fréquence d'échantillonnage du signal (généralement 16000 Hz)

        Returns:
            Signal augmenté de même shape que l'entrée
        """
        # --- Vérification des conditions d'activation ---
        if not self.enabled:
            return audio_array  # Augmentation désactivée globalement

        if random.random() > self.prob:
            return audio_array  # Cet exemple n'est pas augmenté (tirage aléatoire)

        # --- Normalisation du signal en float32 ---
        audio = audio_array.astype(np.float32)

        # --- Étape 1 : Filtre passe-bande 300–3400 Hz ---
        # Simule la bande passante limitée d'une ligne téléphonique analogique
        audio = self.bandpass(samples=audio, sample_rate=sample_rate)

        # --- Étape 2 : Bruit additif gaussien (SNR aléatoire dans [5, 15] dB) ---
        # Simule le bruit de quantification et les interférences de ligne
        audio = self.noise_aug(samples=audio, sample_rate=sample_rate)

        # --- Étape 3 : Simulation codec G.711 (16k → 8k → 16k) ---
        # La dégradation à 8 kHz simule la compression codec téléphonique (µ-law / A-law)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # → [1, N] pour torchaudio

        # Rééchantillonnage vers 8 kHz (perte des hautes fréquences)
        resample_8k  = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
        audio_tensor = resample_8k(audio_tensor)

        # Rééchantillonnage retour vers 16 kHz (requis par Whisper)
        resample_16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=sample_rate)
        audio_tensor = resample_16k(audio_tensor)

        # Retour en numpy 1D, même shape que l'entrée
        return audio_tensor.squeeze(0).numpy()
