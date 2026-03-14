# Segmentation automatique de lesions cutanees en imagerie dermoscopique

**Projet Stat'App — ENSAE Paris (2025)**

Segmentation de lesions cutanees sur le jeu de donnees [ISIC 2016](https://challenge.isic-archive.com/landing/2016/), en progressant des methodes classiques a l'apprentissage profond.

## Resultats

| Methode | DSC | IoU | Test set |
|---------|-----|-----|----------|
| K-means LAB | 0.648 | 0.561 | ISBI 2016 (379 images) |
| CNN Patch (P=32) | 0.760 | 0.639 | ISBI 2016 (379 images) |
| U-Net baseline | 0.863 | 0.781 | ISBI 2016 (379 images) |
| **U-Net final** | **0.882** | **0.807** | ISBI 2016 (379 images) |

Le U-Net final se positionnerait au ~6e rang sur 28 equipes du challenge ISBI 2016.

## Structure du repo

```
statapp/
├── notebooks/                 # Notebooks d'experimentation (numerotes)
│   ├── 01_description         # Analyse exploratoire (EDA)
│   ├── 02_kmeans_segmentation # K-means dans 4 espaces couleur
│   ├── 03_kmeans_optimal_k    # Estimation du K optimal
│   ├── 04_cnn_patch           # CNN par patches (P=32, 48, 64)
│   ├── 05_cnn_patch_complement# Evaluations complementaires CNN
│   ├── 06_hyperparam_unet     # Etude d'ablation U-Net (13 axes, 24 exp.)
│   ├── 07_unet_evaluation     # Evaluation sur test set officiel
│   ├── 08_resolution384_bs8   # Deconfondage resolution vs batch size
│   ├── 09_wilcoxon_tests      # Tests de significativite statistique
│   ├── 10_contrast_subgroups  # Analyse par sous-groupes de contraste
│   └── 11_skip_ablation       # Ablation skip connections (donnees completes)
├── results/                   # Resultats JSON (metriques, per-image)
├── models/                    # Configs JSON des 27 experiences d'ablation
├── docs/overleaf/             # Memoire LaTeX + 15 figures PDF
└── requirements.txt
```

## Methodes

1. **K-means** — Segmentation non supervisee dans l'espace CIE LAB, avec heuristique de luminosite et post-traitement par composante connexe maximale.

2. **CNN Patch** — Classification pixel par pixel via patches 32x32, avec inference par fenetre glissante. Introduit convolutions apprises et supervision.

3. **U-Net** — Architecture encodeur-decodeur avec skip connections. Etude d'ablation sur 13 hyperparametres. Modele final entraine avec Dice loss sur 900 images.

## Donnees

[ISIC 2016 — ISBI 2016 Part 1](https://challenge.isic-archive.com/landing/2016/) : 900 images dermoscopiques d'entrainement + 379 images de test, avec masques de segmentation annotes par des dermatologues.

## Installation

```bash
pip install -r requirements.txt
```

Les notebooks necessitent un GPU pour l'entrainement des modeles deep learning (U-Net, CNN Patch). Les experiences ont ete realisees sur GPU NVIDIA Tesla T4 (16 Go VRAM) via [Onyxia](https://www.onyxia.sh/).

## Auteurs

- Louis Maurice
- Ziad Dira
- Mehdi Sfar
- Clement Hadji

Sous la direction d'Aline Helburg — Headmind Partners
