# ECE ML Hackathon 2026

## Objectif

Le projet prédit l'intensité d'une séance sportive pour un utilisateur. Chaque séance appartient à l'une des trois classes suivantes :

- `easy`
- `moderate`
- `hard`

La métrique utilisée pour la validation locale est le **Macro-F1**. Cette métrique calcule le F1 de chaque classe puis en fait la moyenne, afin de donner le même poids à `easy`, `moderate` et `hard`.

## Données

Les fichiers nécessaires au fonctionnement final du projet doivent être placés dans `src/hackathon_ml/data/` :

```text
src/hackathon_ml/data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Le fichier d'entraînement contient la colonne cible `intensity`. Le fichier de test ne contient pas cette colonne. Les colonnes d'identification et de sortie sont :

- identifiant : `workout_id`
- utilisateur : `user_id`
- cible : `intensity`

Les variables brutes utilisées par le pipeline sont :

```text
user_id
sport
gender
year
month
weekday
start_hour_utc
duration_min
distance_km
speed_kmh
max_speed_kmh
speed_std_kmh
altitude_mean_m
altitude_gain_m
altitude_loss_m
n_points
```

## Architecture

```text
.
├── src/
│   └── hackathon_ml/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       └── data/
│           ├── train.csv
│           ├── test.csv
│           └── sample_submission.csv
├── tests/
├── artifacts/
│   ├── model.joblib
│   ├── test_metrics.json
│   └── submission.csv
├── pyproject.toml
└── README.md
```

Le dossier `imports/` peut contenir des expérimentations ou des anciennes versions, mais il n'est pas nécessaire à l'exécution finale. Le code final et les données doivent rester autonomes dans `src/`.

## Fonctionnement du pipeline

### 1. Chargement des données

`config.py` localise les trois CSV et définit les chemins des artefacts. Le pipeline vérifie que `train.csv`, `test.csv` et `sample_submission.csv` existent avant toute exécution.

### 2. Construction des features

Les données d'entraînement et de test sont réunies temporairement, sans utiliser la cible du test, afin de calculer des transformations cohérentes entre les deux jeux de données.

Le pipeline conserve les variables numériques et catégorielles, puis ajoute :

- `denivele_par_km` : dénivelé positif rapporté à la distance ;
- `vmax_sur_vmoy` : vitesse maximale rapportée à la vitesse moyenne ;
- `speed_calc_kmh` : vitesse recalculée à partir de la distance et de la durée ;
- `denivele_par_min` : dénivelé rapporté à la durée ;
- `effort_speed` : vitesse équivalente intégrant une contribution du dénivelé ;
- `month_index` : position temporelle absolue de la séance ;
- `is_weekend` : indicateur de séance le week-end ;
- `hour_sin` et `hour_cos` : représentation cyclique de l'heure de départ ;
- charge mensuelle par utilisateur et par sport : nombre de séances, durée totale et distance totale ;
- pour chaque groupe, des ratios à la moyenne ;
- pour chaque groupe, des z-scores ;
- le rang percentile de chaque séance ;
- le nombre de séances du groupe.

Les groupes utilisés sont :

- utilisateur : `user_id` ;
- utilisateur et sport : `user_id`, `sport` ;
- sport : `sport` ;
- utilisateur, sport et année : `user_id`, `sport`, `year`.

Les colonnes `sport` et `gender` sont ensuite encodées avec `pandas.get_dummies`. `user_id` et `workout_id` ne sont pas fournis directement au modèle comme variables brutes : l'information utilisateur est exploitée à travers les statistiques relatives ci-dessus.

Les valeurs infinies produites par les ratios sont remplacées par des valeurs manquantes afin d'être traitées par le modèle.

### 3. Modèle

Le modèle utilisé est un `VotingClassifier` en mode `soft`. Il combine les probabilités de trois modèles complémentaires :

```text
features préparées
    │
    ├── RandomForestClassifier
    ├── HistGradientBoostingClassifier
    └── XGBClassifier
        │
        ▼
    VotingClassifier(voting="soft")
        │
        ▼
    classe finale : easy / moderate / hard
```

Les modèles sont configurés ainsi :

- `RandomForestClassifier` : 400 arbres, `min_samples_leaf=2`, équilibrage des classes ;
- `HistGradientBoostingClassifier` : `learning_rate=0.05`, 600 itérations maximum, arrêt anticipé ;
- `XGBClassifier` : 500 arbres, `learning_rate=0.05`, objectif multiclasse et méthode `hist`.

Le vote est un **soft voting** : pour chaque séance, les probabilités prédites par les trois modèles sont moyennées. La classe ayant la probabilité moyenne la plus élevée est retenue.

L'intérêt de cette architecture est de limiter la dépendance à un seul algorithme : Random Forest réduit la variance, tandis que les deux modèles de boosting apprennent des relations non linéaires différentes. L'ensemble doit toutefois être évalué avec la même validation que chaque modèle individuel ; la diversité ne garantit pas automatiquement un meilleur score.

L'ensemble est entraîné sur les features construites à partir du train et du test, puis sauvegardé dans :

```text
artifacts/model.joblib
```

L'artefact contient le `VotingClassifier` entraîné ainsi que la liste des colonnes de features utilisées. Cette liste permet de réaligner les colonnes du jeu de test lors de l'export.

### 4. Validation

La commande `test` utilise une validation croisée stratifiée en 5 parties :

```text
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Chaque partie est prédite par un modèle qui n'a pas vu ces lignes pendant son entraînement. Le score final est le Macro-F1 calculé sur l'ensemble des prédictions hors échantillon.

Le rapport détaillé est affiché dans le terminal et sauvegardé dans :

```text
artifacts/test_metrics.json
```

La validation locale mesure donc le **Macro-F1**, et non l'accuracy. Le seuil de `prog_run` correspond également au Macro-F1.

### 5. Export Kaggle

La commande `export` charge le modèle sauvegardé, construit les features du test, prédit les classes, puis remplit la colonne `intensity` de `sample_submission.csv`.

Le fichier généré par défaut est :

```text
artifacts/submission.csv
```

Avant l'écriture, le pipeline vérifie que le fichier possède les mêmes colonnes et le même nombre de lignes que le fichier d'exemple.

## Installation avec uv

Depuis la racine du projet :

```bash
uv sync
```

## Commandes

### Entraîner le modèle

```bash
uv run hackathon-test train
```

Cette commande entraîne le modèle sur toutes les données d'entraînement et crée `artifacts/model.joblib`.

### Tester le modèle

```bash
uv run hackathon-test test
```

Cette commande lance la validation croisée en 5 parties et affiche le Macro-F1 ainsi que le rapport de classification.

### Exporter une soumission

Le modèle doit d'abord avoir été entraîné :

```bash
uv run hackathon-test train
uv run hackathon-test export
```

Pour choisir un autre chemin de sortie :

```bash
uv run hackathon-test export --output submissions/submission.csv
```

### Exécuter le processus complet

`prog_run` valide d'abord la configuration avec une validation croisée, compare le Macro-F1 au seuil demandé, puis entraîne le modèle final sur toutes les données d'entraînement et exporte la soumission. Le seuil est journalisé comme un contrôle de performance, mais ne bloque pas la sauvegarde : le modèle et la soumission sont conservés même si le score est inférieur.

Par défaut, `prog_run` utilise le seuil de référence du Gate 3, soit un Macro-F1 à `0.66` :

```bash
uv run hackathon-test prog_run --threshold 0.66
```

Le seuil peut être changé manuellement :

```bash
uv run hackathon-test prog_run --threshold 0.60
```

Pour exporter malgré un score inférieur au seuil :

```bash
uv run hackathon-test prog_run \
    --threshold 0.66 \
    --allow-below-threshold \
    --output submissions/submission.csv
```

Le seuil est un **contrôle de qualité**. Il ne force pas le modèle à obtenir ce score et ne modifie pas ses prédictions. L'option `--allow-below-threshold` est conservée pour expliciter l'intention dans les scripts, mais l'export est désormais sauvegardé même sans cette option.

L'ordre exact est :

```text
1. Validation croisée du VotingClassifier
2. Comparaison du Macro-F1 avec le seuil
3. Entraînement final sur 100 % du train, quel que soit le résultat du seuil
4. Sauvegarde de artifacts/model.joblib
5. Export de artifacts/submission.csv
```

Si le seuil n'est pas atteint, un avertissement est écrit dans le terminal et dans le fichier de logs, mais le modèle et la soumission sont tout de même produits.

## Tests techniques

Les tests vérifient notamment la structure du projet, le schéma des données et le contrat minimal du modèle :

```bash
uv run pytest -q
```

Ces tests ne remplacent pas la validation métier `uv run hackathon-test test`, qui mesure les performances sur le dataset.

## Suivre l'exécution avec les logs

Toutes les commandes écrivent leurs messages à la fois dans le terminal et dans :

```text
artifacts/hackathon.log
```

Pour suivre la progression dans un second terminal pendant une exécution longue :

```bash
tail -f artifacts/hackathon.log
```

Le fichier indique notamment :

- le début du chargement des données ;
- la construction des features ;
- le début et la fin de l'entraînement ;
- le début et la fin de chaque fold de validation ;
- le Macro-F1 de chaque fold et le score global ;
- la comparaison avec le seuil de `prog_run` ;
- la sauvegarde du modèle et de la soumission.

Exemple :

```text
2026-10-28 18:20:01 | INFO | TEST: fold 1/5 démarré (18182 entraînement, 4546 validation)
2026-10-28 18:21:14 | INFO | TEST: fold 1/5 terminé | Macro-F1 = 0.6123
```

## Résumé du flux complet

```text
train.csv + test.csv
          │
          ▼
Construction des features relatives à l'utilisateur
          │
          ▼
VotingClassifier soft
    ├── Random Forest
    ├── HistGradientBoosting
    └── XGBoost
          │
          ├── train : artifacts/model.joblib
          │
          ├── test : validation croisée + Macro-F1
          │             └── artifacts/test_metrics.json
          │
          └── export : prédictions du test
                        └── artifacts/submission.csv
```

## Équipe

| Membres | Responsabilités |
| --- | --- |
| 2 membres | Pipeline et préparation des données |
| 2 membres | Modèle, entraînement et validation |
| 2 membres | Documentation, vérification et soumission |
