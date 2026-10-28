# Submission log, team Equipe__

Codes participants : ____ (jamais de nom, d'e-mail ni de téléphone)

Seuils : G1 = ____ (15h50) · G2 = ____ (17h35) · G3 = ____ (17h35)

One line per Kaggle submission. Fill it **before** you look at the score, except the last column.

<!-- SUBS:START -->
| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 15:41 | ____ | Ensemble glouton HGB + HGB lent + ExtraTrees + k-NN (moyenne de probabilités, sans poids de classe), features relatives à la personne + ligne de base locale ±2 mois (sub_1541_v3v4-extratrees+hgb+hgb_slow+knn.csv) | intensité = tiers par personne, la forme évolue dans le temps ; familles aux erreurs différentes | 0.614 |  |
| 16:08 | ____ | Stacking : régression logistique sur les probabilités OOF de 13 modèles (HGB, HGB lent, ExtraTrees, RF, k-NN, logreg ; features v3, v4, v6 avec pentes par personne) (sub_1608_stack-stack.csv) | les modèles se trompent différemment ; le méta-modèle apprend quand faire confiance à qui, mieux qu'une moyenne | 0.627 |  |
<!-- SUBS:END -->

## Familles de modèles comparées en validation (macro-F1 out-of-fold, même CV pour tous)

| Modèle | Famille | macro-F1 OOF | Écart entre plis | Remarque |
| --- | --- | --- | --- | --- |
| LogisticRegression (C=0,5) | linéaire | 0,549 | 0,005 | features standardisées |
| KNeighbors (k=31, distance) | k-NN | 0,569 | 0,004 | |
| RandomForest (400 arbres) | arbres en bagging | 0,599 | 0,004 | |
| ExtraTrees (500 arbres) | arbres aléatoires | 0,603 | 0,002 | le plus stable |
| HistGradientBoosting | boosting | 0,604 | 0,007 | meilleur modèle seul |
| Ensemble HGB + 2×ExtraTrees + poids de classe | ensemble | 0,615 | | sélection gloutonne sur les OOF |

## Schéma de validation

- Utilisateurs du test présents dans le train : 100 % (183 sur 183) → schéma retenu : StratifiedKFold 5 plis, encodage cible par personne refit dans chaque pli (TargetEncoder cross-fitted)
- Labels = tiers par personne (écart-type des parts de classe 0,040 contre 0,068 attendu au hasard), mais pas par année ni par mois (1,4 × le hasard) : la forme évolue dans le temps
- Corrélation CV / public : ____

## Gate 2 : matrice de confusion et analyse d'erreur

![confusion](out/figures/confusion_blend.png)

Phrase d'analyse : ____

## Idées (hypothèse, prédiction, résultat)

| Heure | Idée | Pourquoi | Prédiction | Résultat CV | Gardée ? |
| --- | --- | --- | --- | --- | --- |
| 15:00 | Features relatives à la personne (z-score, ratio, rang par personne et par personne × sport) | « hard » est relatif à chacun | HGB > logreg brut | HGB 0,599 / logreg 0,541 | oui |
| 15:05 | Quotas d'un tiers par (personne, sport) en post-traitement | labels = tiers par personne (mesuré) | +0,01 | +0,000 | non |
| 15:16 | Ligne de base locale : séance vs mêmes séances de la personne à ±2 mois | la forme progresse (début d'historique plus dur, mesuré) | +0,01 | HGB 0,599 → 0,604 | oui |
| 15:22 | Plus de capacité : HGB lent (63 feuilles, pas 0,03, 2 000 itérations, 50 % des features par coupure) | tester si le modèle manque de capacité | +0,003 | 0,604 → 0,6085 | oui (dans l'ensemble) |
| 15:24 | Ensemble glouton de familles (moyenne de probabilités sur les OOF) | erreurs différentes entre boosting et arbres aléatoires | +0,005 | 0,6085 → 0,614 | oui |
| 15:26 | v4 : vitesse équivalente-effort (100 m de montée = 1 km), heure relative à l'habitude de la personne, week-end | à vitesse égale, plus de dénivelé = plus dur ; week-end plus dur (mesuré) | +0,003 | HGB 0,6044 → 0,6064 | oui |
| 15:43 | v5 : ligne de base sur ±1 et ±4 mois, charge des 3 mois précédents | fatigue et forme à plusieurs échelles | +0,003 | HGB 0,6064 → 0,6030 | non (retiré) |
| 15:53 | v6 : pentes par personne (ridge par personne × sport rétrécie vers le modèle global, cross-fitted) | chaque personne a sa propre sensibilité à la vitesse, la durée, le dénivelé | linéaire : corrélation 0,41 → 0,54 | linéaire seul : 0,499 → 0,562 ; logreg 0,549 → 0,582, ExtraTrees 0,606 → 0,611, HGB 0,606 → 0,609 | oui |
| 16:05 | Stacking : régression logistique sur les log-probabilités OOF de tous les modèles (C choisi en CV) | les modèles se trompent différemment, un méta-modèle pondère mieux qu'une moyenne | +0,005 | 0,617 → 0,627 | oui |

## Meilleure idée (pour le jury)

____
