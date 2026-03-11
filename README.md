# StatApp — Segmentation de lésions cutanées (U-Net / ISIC 2016)

Étude d'hyperparamètres d'un U-Net pour la segmentation de lésions cutanées en dermoscopie.

## Structure du projet

```
statapp/
  dataset_ISIC/           # Dataset (non inclus, voir ci-dessous)
  models/                 # Modèles entraînés (.keras) + résultats (.json)
  docs/                   # Documentation des choix techniques
  screenshots/            # Captures d'écran
  archive/                # Anciens fichiers (v1)
  description.ipynb       # Analyse exploratoire du dataset
  hyperparam_unet_v2.ipynb  # Étude d'hyperparamètres (13 exp + modèle final)
  unet_isic2016.ipynb     # Implémentation de base du U-Net
```

## Setup

### 1. Dataset

Télécharger le dataset ISIC 2016 (Task 1 — Segmentation) :

- **Kaggle** : https://www.kaggle.com/datasets/mahmudulhasantasin/isic-2016-original-dataset
- **ISIC Archive (officiel)** : https://challenge.isic-archive.com/data/

Extraire le contenu dans `dataset_ISIC/` pour obtenir :
```
dataset_ISIC/
  ISBI2016_ISIC_Part1_Training_Data/       # 900 images (.jpg)
  ISBI2016_ISIC_Part1_Training_GroundTruth/  # 900 masques (.png)
```

### 2. Dépendances

```bash
pip install -r requirements.txt
```

### 3. Entraînement

Ouvrir `hyperparam_unet_v2.ipynb` et exécuter toutes les cellules. Le notebook :
- Entraîne 13 expériences d'hyperparamètres sur 50% du dataset
- Sélectionne automatiquement les meilleurs paramètres
- Entraîne le modèle final sur 100% du dataset
- Optimise le seuil de binarisation
- Génère des visualisations Grad-CAM

Les modèles et résultats sont sauvegardés dans `models/`. Le cache permet de relancer le notebook sans tout ré-entraîner.

## Méthodologie

**Passe 1** : on fixe une baseline et on varie un seul hyperparamètre à la fois (13 expériences).

**Passe 2** : on combine les meilleurs choix pour construire le modèle final.

Détails dans `docs/choix_modele_final.txt`.
