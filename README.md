###########################################
# DIT :: Deep Learning - CNN Cats vs Dogs #
###########################################

# Objectif du projet

Ce projet vise à comparer deux approches de classification d’images pour distinguer **chats** et **chiens** à partir du dataset *Dogs vs Cats*.
Deux expériences ont été menées :

- **Expérience A : CNN from scratch**
  > Conception d’un réseau convolutionnel simple (3 blocs Conv-BN-Pool + Dropout)

- **Expérience B : Transfert Learning**
  > Utilisation d’un modèle pré-entraîné (**ResNet18**) sur ImageNet, adapté à 2 classes

---

## Structure du dépôt

```
cnn-catsdogs-InzaBamba/
├── Loading_Image_Data.ipynb      # Notebook principal (Colab)
├── requirements.txt              # Dépendances du projet
├── .gitignore                    # Fichiers à exclure du dépôt
├── checkpoints/                  # Sauvegardes locales des modèles (.pth)
├── data/                         # (Non inclus) dossier contenant les images
└── README.md
```

### `.gitignore` minimal
```gitignore
data/
checkpoints/
*.pt
*.pth
runs/
__pycache__/
.ipynb_checkpoints/
```

---

##  Environnement & installation

### Option 1 — via pip

```bash
pip install -r requirements.txt
```

### Option 2 — via Colab

Le projet est conçu pour être exécuté sur **Google Colab** (avec GPU T4 activé) :

1. Ouvre le notebook dans Colab  
2. Monte ton Google Drive ou Kaggle dataset  
3. Exécute les cellules pas à pas

---

## Téléchargement et organisation des données

###  Dataset : *Dogs vs Cats*

Télécharge depuis [Kaggle – Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)

#### Étapes :
1. Crée un dossier `data/` à la racine du projet  
2. Place les images dans la structure suivante :

```
data/
 ├─ train/
 │   ├─ cat/
 │   └─ dog/
 └─ test/
     ├─ cat/
     └─ dog/
```

>  Le dossier `data/` **ne doit pas être poussé sur GitHub** (déjà exclu dans `.gitignore`).

---

##  Reproduction des expériences

###  1. Expérience A — CNN from scratch

Architecture :
- 3 blocs convolutionnels  
- Batch Normalization + Dropout  
- Optimiseur : Adam (`lr=1e-3`)  
- Scheduler : StepLR (`step_size=4`, `gamma=0.5`)

Entraînement :
```python
set_seed(42)
model = CNNFromScratch(num_classes=2).to(device)
hist_adam = run_training(model, train_loader, test_loader,
                         optimizer_name='adam', lr=1e-3,
                         epochs=10, use_scheduler=True,
                         ckpt_name="cnn_adam.pth")
```

Chargement du modèle :
```python
best_model = CNNFromScratch(num_classes=2).to(device)
best_model.load_state_dict(torch.load("checkpoints/cnn_adam.pth"))
```

---

###  2. Expérience B — Transfert Learning (ResNet18)

Architecture :
- Backbone : ResNet18 pré-entraîné sur ImageNet  
- Couches gelées (`freeze=True`)  
- Nouveau classifieur : `Linear(512→256→2)` + Dropout

Entraînement :
```python
set_seed(42)
model_resnet = get_resnet18(num_classes=2, pretrained=True, freeze=True).to(device)
hist_resnet = run_training(model_resnet, train_loader, test_loader,
                           optimizer_name='adam', lr=5e-4,
                           epochs=10, use_scheduler=True,
                           ckpt_name="resnet_transfer.pth")
```

Chargement du modèle :
```python
best_model = get_resnet18(num_classes=2, pretrained=True, freeze=True).to(device)
best_model.load_state_dict(torch.load("checkpoints/resnet_transfer.pth"))
```

---

##  Résultats & Analyse

| Expérience | Modèle | Optimiseur | Accuracy (Test) | Précision | Rappel | Observations |
|-------------|---------|-------------|------------------|------------|---------|---------------|
| A | CNN from scratch | Adam | ~0.80 | ~0.80 | ~0.80 | Bonne convergence mais surapprentissage |
| B | ResNet18 (Transfer) | Adam | ~0.98 | ~0.98 | ~0.98 | Excellente généralisation et convergence rapide |

**Courbes d’apprentissage :**
- Expérience A : légère divergence entre train/val après 5 époques  
- Expérience B : convergence stable et meilleure robustesse  

**Conclusion :**
Le modèle **ResNet18 Transfer Learning** offre de meilleures performances et une convergence plus rapide que le CNN from scratch.



## Persistance du modèle

Les modèles sauvegardés sont enregistrés localement dans :
```
checkpoints/
 ├─ cnn_adam.pth
 └─ resnet_transfer.pth
```
---

## Auteurs

Projet réalisé par **Inza BAMBA**  
Encadré dans le cadre du module *Deep Learning – CNN Cats vs Dogs*

---

## Licence

Ce projet est distribué sous licence MIT – vous pouvez le réutiliser librement à des fins éducatives.
