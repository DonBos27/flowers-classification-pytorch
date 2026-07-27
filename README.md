# Classification d'Images de Fleurs par Deep Learning avec PyTorch

## 1. Présentation du projet

Ce projet a été réalisé dans le cadre du cours de **Deep Learning** du Master 1 en Intelligence Artificielle.

L'objectif est de développer une application capable de reconnaître automatiquement l'espèce d'une fleur à partir d'une image en utilisant les techniques de **Transfer Learning** avec PyTorch.

Le modèle est entraîné sur le célèbre jeu de données **102 Category Flower Dataset** et permet ensuite de prédire les cinq espèces les plus probables pour une nouvelle image.

---

## 2. Objectifs

Les principaux objectifs de ce projet sont les suivants :

- Prétraiter les images pour l'apprentissage profond ;
- Construire un pipeline complet de classification d'images ;
- Exploiter un modèle pré-entraîné (Transfer Learning) ;
- Entraîner un nouveau classifieur sur le jeu de données des fleurs ;
- Évaluer les performances du modèle ;
- Sauvegarder le modèle entraîné sous forme de checkpoint ;
- Développer un programme permettant de prédire automatiquement la classe d'une nouvelle image.

---

## 3. Jeu de données

Le projet utilise le **102 Category Flower Dataset**.

Le jeu de données est organisé en trois ensembles :

| Ensemble | Nombre d'images |
|-----------|----------------:|
| Entraînement | 6552 |
| Validation | 818 |
| Test | 819 |

Chaque image appartient à l'une des **102 catégories de fleurs**.

Le fichier **cat_to_name.json** permet d'associer chaque identifiant numérique au nom réel de la fleur.

---

## 4. Technologies utilisées

- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow (PIL)
- VS Code

---

## 5. Structure du projet

```
Bosenga_Don_master1_ia_oct2025_1/

│
├── flowers/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── checkpoints/
│   └── flower_classifier_checkpoint.pth
│
├── Image Classifier Project.ipynb
├── train.py
├── predict.py
├── cat_to_name.json
└── README.md
```

---

## 6. Prétraitement des données

Avant l'entraînement, plusieurs transformations sont appliquées aux images.

### Entraînement

- Redimensionnement
- Recadrage aléatoire
- Rotation aléatoire
- Retournement horizontal
- Conversion en tenseur
- Normalisation

### Validation et Test

- Redimensionnement
- Recadrage centré
- Conversion en tenseur
- Normalisation

Ces opérations permettent d'améliorer la capacité de généralisation du modèle.

---

## 7. Architecture du modèle

Le projet utilise le **Transfer Learning**.

Le modèle choisi est :

**ResNet18 pré-entraîné sur ImageNet.**

Les couches de convolution sont gelées (leurs poids ne sont pas modifiés).

Seul le classifieur final est remplacé par un nouveau réseau de neurones adapté aux **102 classes** du jeu de données.

---

## 8. Entraînement du modèle

Pendant l'entraînement, les éléments suivants sont utilisés :

- Fonction de coût : CrossEntropyLoss
- Optimiseur : Adam
- Learning Rate : configurable
- Nombre d'époques : configurable

À chaque époque, le modèle est évalué sur l'ensemble de validation.

Le meilleur modèle est automatiquement sauvegardé.

---

## 9. Sauvegarde du modèle

Une fois l'entraînement terminé, un checkpoint est enregistré.

Celui-ci contient notamment :

- l'architecture du modèle ;
- les poids entraînés ;
- le dictionnaire des classes ;
- les paramètres du classifieur.

Ce fichier permet de recharger le modèle sans avoir à le réentraîner.

---

## 10. Prédiction

Le script `predict.py` permet de :

- charger le checkpoint ;
- reconstruire le modèle ;
- prétraiter une nouvelle image ;
- calculer les probabilités de chaque classe ;
- afficher les **Top-K prédictions** avec leurs probabilités.

---

## 11. Résultats

Au cours des expérimentations, le modèle a progressivement amélioré ses performances sur l'ensemble de validation.

Les courbes de perte (*Loss*) et de précision (*Accuracy*) montrent une convergence satisfaisante du modèle.

Les prédictions obtenues démontrent la capacité du réseau à identifier correctement plusieurs espèces de fleurs.

---

## 12. Lancement du projet

### Entraîner le modèle

```bash
python train.py flowers \
--arch resnet18 \
--epochs 5 \
--learning_rate 0.001 \
--hidden_units 512 \
--save_dir checkpoints \
--gpu
```

### Effectuer une prédiction

```bash
python predict.py \
flowers/test/1/image_06743.jpg \
checkpoints/flower_classifier_checkpoint.pth \
--top_k 5 \
--category_names cat_to_name.json \
--gpu
```

---

## 13. Perspectives d'amélioration

Plusieurs améliorations peuvent être envisagées :

- Tester d'autres architectures (ResNet50, EfficientNet, Vision Transformer) ;
- Réaliser une recherche automatique des hyperparamètres ;
- Développer une interface graphique ;
- Déployer le modèle sous forme d'application Web ou mobile.

---