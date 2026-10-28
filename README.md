# 🏆 ECE ML Hackathon 2026 - Équipe XX

## 📌 Objectif

L'objectif de ce hackathon est de prédire la difficulté d'une séance sportive pour un utilisateur à partir des données collectées par une application de suivi sportif.

Les séances doivent être classées dans l'une des trois catégories :

- easy
- moderate
- hard

L'évaluation est réalisée à l'aide du Macro-F1 Score, qui accorde le même poids aux trois classes.

## 👥 Équipe

| Membre | Rôle |
| --- | --- |
| Membre 1 | Pipeline / Data preparation |
| Membre 2 | Pipeline / Feature engineering |
| Membre 3 | Model / Training |
| Membre 4 | Model / Validation & optimization |
| Membre 5 | Docs / Reporting & verification |
| Membre 6 | Docs / Verification & submission |

## 📂 Structure du projet

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── training.ipynb
│
├── src/
│   └── hackathon_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   └── loader.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── training.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   └── train_pipeline.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           └── paths.py
│
├── tests/
│   ├── test_data_schema.py
│   ├── test_model_contract.py
│   └── test_project_structure.py
│
├── LOG.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## 📊 Données utilisées

Chaque séance contient notamment :

- user_id
- sport
- gender
- hour
- duration
- distance
- speed
- max_speed
- elevation_gain

La variable cible est :

- intensity

avec les classes :

- easy
- moderate
- hard

## 🔧 Prétraitement

Les étapes de prétraitement appliquées sont :

### Variables catégorielles

- sport
- gender

Encodage via : `OneHotEncoder`

### Variables numériques

- hour
- duration
- distance
- speed
- max_speed
- elevation_gain

### Gestion des valeurs manquantes

`SimpleImputer(strategy="median")`

## 🧠 Modèle retenu

Nous utilisons un `VotingClassifier` en mode `soft` combinant trois modèles complémentaires :

1. Random Forest
   - robuste au bruit
   - capture les relations non linéaires
   - peu sensible au scaling
   - modèle utilisé : `RandomForestClassifier`

2. XGBoost
   - excellent pouvoir prédictif
   - gestion performante des interactions complexes
   - approche boosting
   - modèle utilisé : `XGBClassifier`

3. HistGradientBoosting
   - très rapide sur grands volumes de données
   - implémentation moderne du Gradient Boosting
   - bonne généralisation
   - modèle utilisé : `HistGradientBoostingClassifier`

### Ensemble Learning

Les probabilités fournies par chaque modèle sont combinées via :

```python
VotingClassifier(voting="soft")
```

### Architecture

```text
+----------------+      +----------------+      +----------------+
| Random Forest  | ---> |    XGBoost     | ---> | HistGradient  |
+----------------+      +----------------+      |   Boosting     |
                                                  +----------------+
                                                          |
                                                          v
                                               Soft Voting Ensemble
                                                          |
                                                          v
                                                  Prediction Finale
```

## ⚙️ Pipeline

```text
Preprocessing
    ↓
Feature Engineering
    ↓
Random Forest
    ↓
XGBoost
    ↓
HistGradientBoosting
    ↓
Soft Voting
    ↓
Prediction
```

## 📈 Validation

Pour évaluer nos modèles, nous utilisons :

- `StratifiedKFold` ou validation croisée stratifiée
- métrique : `f1_macro`

Exemple :

```python
cross_val_score(
    model,
    X,
    y,
    scoring="f1_macro",
    cv=5,
)
```

## 🚀 Lancement

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Entraînement

```bash
python src/train.py
```

### Génération des prédictions

```bash
python src/predict.py
```

### Création du fichier Kaggle

Fichier produit : `submission.csv`

Format attendu :

```csv
workout_id,intensity
123,easy
124,moderate
125,hard
```

## 💡 Pistes d'amélioration

- Feature engineering sur l'heure (`morning`, `afternoon`, `evening`)
- Statistiques par utilisateur (`user_id`)
- Ratio vitesse / vitesse maximale
- Distance par durée
- Features cycliques sur l'heure (`sin`, `cos`)
- Optimisation des hyperparamètres avec `Random Search` ou `Optuna`
- Pondération des modèles dans le `VotingClassifier`

## 📋 Résultats

| Modèle | Macro F1 CV |
| --- | --- |
| Random Forest | TBD |
| XGBoost | TBD |
| HistGradientBoosting | TBD |
| Voting Classifier | TBD |

## 🏁 Conclusion

Nous avons construit un modèle d'ensemble basé sur la complémentarité de :

- Random Forest
- XGBoost
- HistGradientBoosting

afin d'obtenir une meilleure généralisation qu'un modèle individuel. Le choix du `Soft Voting` permet d'exploiter les probabilités prédites par chaque algorithme et d'améliorer la robustesse des prédictions sur les trois classes du challenge.