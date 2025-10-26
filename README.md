# Signify - Reconnaissance de la Langue des Signes en Temps Réel

## 🎯 Objectif
Développer une application capable de reconnaître des gestes de la Langue des Signes (ASL) en temps réel via webcam et de les convertir en texte/voix.

## 🏗️ Architecture
- **Détection** : MediaPipe pour la détection des mains
- **Modèle** : CNN (ResNet18) pour la classification des gestes
- **Interface** : OpenCV pour l'affichage temps réel
- **Synthèse vocale** : pyttsx3 pour la conversion texte → voix

## 📁 Structure du projet
```
signify/
├── data/                    # Dataset ASL
├── checkpoints/             # Modèles entraînés
├── src/
│   ├── train/              # Scripts d'entraînement
│   ├── inference/          # Scripts d'inférence
│   └── utils/              # Utilitaires
├── requirements.txt         # Dépendances
└── README.md
```

## 🚀 Installation
```bash
# Cloner le projet
git clone <votre-repo>
cd signify

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Dataset
- **Source** : ASL Alphabet Dataset (Kaggle)
- **Classes** : A-Z + del + space + nothing (29 classes)
- **Images** : ~87k images, 3000 par classe
- **Split** : 70% train, 15% val, 15% test

## 🧠 Modèle
- **Architecture** : ResNet18 pré-entraîné
- **Input** : Images 96x96 RGB
- **Output** : 29 classes ASL
- **Performance** : >95% accuracy

## 🎮 Utilisation
```bash
# Entraînement
python -m src.train.train_model --data data/split --epochs 30

# Évaluation
python -m src.train.eval_model --ckpt checkpoints/best_model.pt

# Inférence temps réel
python -m src.inference.cam_infer --ckpt checkpoints/best_model.pt
```

## 🔧 Fonctionnalités
- ✅ Détection de main en temps réel
- ✅ Classification des gestes ASL
- ✅ Stabilisation temporelle
- ✅ Interface utilisateur intuitive
- ✅ Synthèse vocale
- ✅ Export ONNX pour déploiement

## 📈 Performance
- **Accuracy** : >95% sur test set
- **FPS** : 30+ en temps réel
- **Latence** : <100ms par prédiction