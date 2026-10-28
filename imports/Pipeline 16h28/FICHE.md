# Fiche d'explication (à recopier sur papier, appareils fermés)

## 1. Le problème
Prédire l'intensité d'une séance (easy / moderate / hard), métrique macro-F1. 183 personnes, toutes présentes dans le train et le test.

## 2. Ce que nous avons mesuré avant de modéliser
- Les classes sont des **tiers par personne** : la part de chaque classe varie très peu d'une personne à l'autre
  (écart-type 0,040 contre 0,068 attendu au hasard). « Hard » veut dire « dur pour cette personne ».
- Mais pas des tiers par mois ou par année (1,4 fois le hasard) : **la forme évolue**. Les premières séances
  de l'historique sont plus souvent dures que les dernières (on progresse, le même effort devient plus facile).
- La vitesse brute prédit peu (corrélation 0,11), la vitesse **par rapport aux habitudes de la personne** beaucoup plus (0,31).

## 3. Validation
StratifiedKFold à 5 plis (les personnes du test sont toutes dans le train, donc on mélange les séances, pas les personnes).
Tout ce qui utilise la cible (encodage par personne, pentes par personne) est recalculé **dans chaque pli**
(cross-fitting) : jamais de fuite de la réponse. On choisit sur le macro-F1 out-of-fold, pas sur le leaderboard.

## 4. Les features
1. **Relatives à la personne** : z-score, ratio et rang de chaque mesure (vitesse, durée, distance, dénivelé)
   par rapport à la personne, à la personne dans ce sport, et dans cette année.
2. **Ligne de base locale** : la séance comparée aux séances de la même personne dans le même sport à ±2 mois
   (suit la progression de la forme).
3. **Effort** : vitesse équivalente (100 m de montée = 1 km de plat), heure relative à l'heure habituelle de la personne, week-end.
4. **Encodage cible par personne** (part de chaque classe chez la personne, dans ce sport, ce mois), cross-fitted.
5. **Pentes par personne** : une petite régression par personne et par sport, rétrécie vers la régression commune,
   car chacun réagit différemment (la durée pèse plus pour l'un, la vitesse pour l'autre).

## 5. Les modèles comparés (même validation)
| Modèle | macro-F1 OOF |
| --- | --- |
| Régression logistique | 0,549 |
| k plus proches voisins | 0,569 |
| Random Forest | 0,599 |
| ExtraTrees | 0,603 |
| HistGradientBoosting | 0,604 |
| Ensemble (moyenne des probabilités) | 0,615 |

Final : ____ (à compléter).

## 6. Analyse d'erreur
La classe du milieu (moderate) est la plus difficile : elle se confond avec ses deux voisines ; easy et hard
se confondent rarement entre elles. ____ (chiffres de la matrice de confusion finale).

## 7. Meilleure idée
Comprendre que l'intensité est **relative à la personne et au moment** : comparer chaque séance aux habitudes
de la personne, à sa forme du moment, et apprendre la sensibilité propre à chacun.
