# Signify - Reconnaissance de la Langue des Signes en Temps Réel

## Objectif
Développer une application capable de reconnaître des gestes de l'alphabet de la Langue des Signes (ASL) en temps réel via webcam.


## Installation
```bash
# Cloner le projet
git clone https://github.com/Victoooooooor/signify.git
cd signify

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## Dataset 1
- **Source** : ASL Alphabet Dataset 
- **Lien** : https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Classes** : A-Z supprimer : del + space + nothing 
- **Images** : 3000 par classe

## Dataset 2
- **Source** : American Sign Language Dataset
- **Lien** : https://www.kaggle.com/datasets/ayuraj/asl-dataset
- **Classes** : A-Z supprimer : 1-9 
- **Images** : 70 par classe


## Utilisation
```bash
# Entraînement
python -m src.train.train_model --data data/split --epochs 30

# Évaluation
python -m src.train.eval_model --ckpt checkpoints/best_model.pt

# Inférence temps réel
python -m src.inference.cam_infer --ckpt checkpoints/best_model.pt
```
