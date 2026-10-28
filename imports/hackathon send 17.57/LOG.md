# Submission log, team Equipe_06

Codes participants : ____ (jamais de nom, d'e-mail ni de téléphone)

Seuils : G1 = ____ (15h50) · G2 = ____ (17h35) · G3 = ____ (17h35)

One line per Kaggle submission. Fill it **before** you look at the score, except the last column.

<!-- SUBS:START -->
| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 16:08 | ____ | Stacking : régression logistique sur les probabilités OOF de 13 modèles (HGB, HGB lent, ExtraTrees, RF, k-NN, logreg ; features v3, v4, v6 avec pentes par personne) (sub_1608_stack-stack.csv) | les modèles se trompent différemment ; le méta-modèle apprend quand faire confiance à qui, mieux qu'une moyenne | 0.627 | 0.64276 |
| 17:01 | ____ | Même stacking que 16:08, puis affectation sous contrainte : par personne, au plus ceil(N/3) séances de chaque classe sur tout l'historique (sub_1701_s1608-quota.csv) | mesuré : les labels sont des tiers exacts par personne sur train + test (variance 1,12 x tirage sans remise, 0,28 x hasard) ; on impose ces comptes au lieu de l'argmax | 0.6285 | 0.65118 |
| 17:21 | ____ | Stacking élargi à 24 modèles (13 de la 16:08 + v6 HGB lent, ET graine 2, HGB max_features 0,3, ordinal ; v7 HGB lent, ET, HGB mf3, k-NN par personne, logreg, ordinal), puis quotas de tiers par personne (sub_1720_c4-quota-caps.csv) | les nouveaux modèles n'aidaient pas l'argmax (0,6273 contre 0,6271) mais leurs probabilités servent mieux l'affectation sous quotas : 0,6304 -> 0,6325 (validation 20 plis) | 0.6325 | 0.65061 |
| 17:32 | ____ | Même stacking que 16:08 ; les quotas durs sont remplacés par un équilibrage souple par personne : on décale le prior de chaque classe jusqu'à ce que la somme des probabilités de ses séances inconnues égale les comptes attendus (N/3 - connus), puis argmax (sub_1731_s1608-quota-bal.csv) | les comptes sont tenus en espérance au lieu d'un plafond dur : moins de dégâts quand le modèle est confiant. Validation 20 plis : 0,6304 -> 0,6358 (graine 11), 0,6288 -> 0,6332 (graine 33) | 0.6358 | 0.65472 |
| 17:43 | ____ | Variante de la 17:31 : comptes attendus fractionnaires (N/3 - connus, non arrondis) dans l'équilibrage souple (sub_1735_final-stack-quota.csv) | éviter l'arrondi des comptes attendus ; 0,6368 contre 0,6358 en validation 20 plis (écart dans le bruit) | 0.6368 | 0.65194 |
| 17:52 | ____ | Stacking élargi (24 modèles, c4) + équilibrage souple par personne, puis poids de décision 1,2 sur moderate avant l'argmax (sub_1752_c4-quota-bal-mod12.csv) | le macro-F1 récompense la classe faible : pousser moderate après l'équilibrage. Validation 20 plis : 0,6388 / 0,6382 (graines 11 / 33) contre 0,6358 / 0,6332 pour la 17:31 ; poids moderate seul sur c4 : +0,0023 / +0,0032 | 0.6388 | 0.65720 |
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
- Corrélation CV / public : stacking 0,627 en CV → 0,643 en public ; + quotas 0,630 (validation 20 plis) → 0,651 en public. Même sens de progrès, écart stable ~+0,02

## Gate 2 : matrice de confusion et analyse d'erreur

![confusion](out/figures/confusion_final_stack_quota.png)

Modèle final : stacking (13 modèles) + équilibrage souple des tiers par personne, comptes entiers (sub_1731, reproduite à l'identique par le notebook ; macro-F1 OOF 0,636 en validation 20 plis, public 0,65472). Lignes = vrai, colonnes = prédit :

| vrai \ prédit | easy | moderate | hard |
| --- | --- | --- | --- |
| easy | 5364 | 1620 | 545 |
| moderate | 1991 | 3868 | 1744 |
| hard | 718 | 1623 | 5255 |

Phrase d'analyse : l'erreur est surtout entre classes voisines. Moderate reste la classe faible (F1 0,53 contre 0,69 pour easy et 0,69 pour hard) ; elle part pour 26 % en easy et 23 % en hard, alors qu'easy et hard ne se confondent que dans 6 % des cas. Le modèle a appris l'ordre, mais la frontière d'un tiers par personne reste floue au milieu : une séance « moyenne » ne se distingue des séances faciles ou dures de la personne que par de petits écarts à ses habitudes. Imposer à chaque personne ses tiers (en espérance) remonte surtout easy et hard (rappels 0.712 et 0.692).

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
| 16:20 | Indice 1 des organisateurs : « difficile est mesuré pour chaque personne, comparer à ses habitudes dans ce sport » | déjà mesuré à 14h50 (tiers par personne, vitesse relative corr. 0,31 contre 0,11 brute) | aucun gain attendu, déjà en place | couvert depuis v2 : z-score/ratio/rang par personne×sport, ligne de base ±2 mois, pentes par personne | déjà intégré |
| 17:00 | Tiers exacts par personne sur tout l'historique : imposer les comptes de classe du test | variance des comptes du train = 1,12 x tirage sans remise depuis des tiers exacts (0,28 x hasard) : le test doit compléter chaque personne à un tiers par classe | +0,002 à +0,005 | 0,6272 -> 0,6285 (5 plis), 0,6298 (10 plis), 0,6302 (20 plis) : le gain croît quand la validation se rapproche du vrai test (moins d'inconnues par personne). Public : 0,64276 -> 0,65118 (+0,008) à modèle identique | oui |
| 17:05 | Variantes des quotas : comptes exacts (min et max), coupe ordinale sur P(moderate) + 2 P(hard) | serrer la contrainte ; coller à la construction des labels (tiers d'une mesure continue) | >= plafonds seuls | plafonds 0,6285 > comptes exacts 0,6276 > ordinal 0,6265 (< argmax 0,6272) : les arrondis rendent l'exact trop rigide, la coupe ordinale jette l'information des probabilités | non, on garde les plafonds |
| 17:08 | Méta-modèle HistGradientBoosting au lieu de la régression logistique du stacking | interactions entre modèles | +0,002 | 0,6213 contre 0,6263 (même CV) : sur-apprend les log-probas, la logreg très régularisée (C = 0,01) reste meilleure | non |
| 17:20 | Quotas sur le stacking élargi (24 modèles, dont v7 et k-NN par personne) | les modèles v7 n'aident pas l'argmax mais diversifient les probabilités que l'affectation exploite | +0,001 | 0,6304 -> 0,6325 en validation 20 plis (base 0,6271 -> 0,6273) ; 96 % de prédictions communes avec la 17:01. Public 0,65061 contre 0,65118 : écart de -0,0006, dans le bruit | non : on garde le modèle plus simple de la 17:01 |
| 17:28 | Fuite d'ordre des `workout_id` | un identifiant attribué dans l'ordre pourrait trahir le temps ou le label | aucune si les ID sont mélangés | corrélation 0,00 avec le temps, la personne et le label ; ID contigus | non (rien à exploiter) |
| 17:28 | Labels des séances voisines (même personne, même jour ou même heure) | une même journée peut partager l'effort | +0,002 | accord 0,50 contre 0,32 au hasard, mais stacking 0,6271 -> 0,6277 : déjà capté par l'encodage par mois | non |
| 17:28 | Critère d'affectation des quotas : log p, p, racine de p, p au carré | la forme du coût change l'affectation | +0,001 | 0,6300 à 0,6306 (référence 0,6304) : équivalents | non, log p gardé |
| 17:28 | Comptes exacts reconstruits (tiers du rang sur tout l'historique, reste attribué à hard puis moderate) | motif sans aucune incohérence sur 20 plis x 183 personnes (16 et 28 pour les autres motifs) | >= plafonds | 0,6292 contre 0,6304 pour les plafonds seuls : trop rigide quand les probabilités se trompent | non |
| 17:30 | Équilibrage souple par personne + argmax, au lieu des quotas durs (trouvé par l'agent idées, revérifié) | tenir les comptes attendus en espérance plutôt qu'en plafond : un plafond force des erreurs quand le modèle est sûr de lui | +0,002 | 20 plis : s1608 0,6304 -> 0,6358 (graine 11), 0,6288 -> 0,6332 (graine 33) ; stack élargi 0,6325 -> 0,6365, 0,6300 -> 0,6350. Équilibrage puis plafonds : pas de gain (le gain vient de l'absence de contrainte dure) | oui |
| 17:30 | Voisins temporels plus larges (mois ± 1) et ajustement par les labels connus du même mois | autocorrélation des labels dans l'historique | +0,002 | séances adjacentes : même label 39 % contre 33 % au hasard, déjà capté ; ajustement 0,6272 -> 0,6215 ; ± 1 mois neutre ; pas de jour dans les données | non |
| 17:40 | Modèle par paires : « laquelle de ces deux séances de la même personne est la plus dure ? », puis comparaison de chaque séance inconnue aux séances connues easy, moderate et hard de la personne (tools/pairwise.py) | colle à la construction des labels (rang dans l'historique de la personne) | +0,002 | seul 0,537 ; dans le stacking 0,6271 -> 0,6278 ; après équilibrage 0,6358 -> 0,6361 : information déjà présente | non |
| 17:42 | Température des probabilités avant l'équilibrage (P^T, T de 0,6 à 1,7) | régler la confiance du modèle avant de décaler les priors | +0,001 | T = 1 optimal ou à égalité (T = 1,3 : +0,0012 sur une graine, -0,0003 sur l'autre) | non, T = 1 |
| 17:45 | Comptes attendus fractionnaires plutôt qu'entiers dans l'équilibrage | éviter l'arrondi | +0,001 | CV 0,6368 contre 0,6358 ; public 0,65194 contre 0,65472 pour les comptes entiers (17:31) : écart de CV dans le bruit, le public tranche pour les entiers | non : comptes entiers (17:31) = modèle final du notebook |
| 17:52 | Stacking s1608 + 2 modèles v7 plus divers (HGB lent graine 2, ExtraTrees 1000 arbres sqrt) | plus de diversité dans le stacking | +0,002 | stacking 0,6253 ; après équilibrage 0,6368 (+0,0010, graine 11) et 0,6352 (+0,0020, graine 33) : sous le seuil sur une graine, et le stack élargi c4 n'a pas suivi en public | non |
| 17:50 | Retrouver la règle de label : un score par séance, coupé en tiers par personne (driver idées 5) | les labels sont des tiers exacts par personne : si le score est une formule des colonnes, on gagne beaucoup | +0,02 | unité de rang = personne sur tout l'historique (ratio de variance 1,12 ; personne×année 18) ; meilleur score simple : vitesse, 0,46 en tiers ; combinaison linéaire par paires 0,46 ; une séance qui domine une autre sur 6 colonnes est moins dure dans 30 % des cas : variable cachée (fréquence cardiaque probable), pas de formule à retrouver | non |
| 17:51 | Poids de décision sur moderate après l'équilibrage (argmax de P × w, w_moderate de 0,9 à 1,35) | le macro-F1 pèse autant la classe faible : la pousser un peu rapporte plus qu'elle ne coûte | +0,002 | s1608 : 1,15 -> 0,6371 / 0,6370, 1,2 -> 0,6370 / 0,6368 (contre 0,6358 / 0,6332), 0,9 et 1,35 en baisse : courbe régulière ; c4 : 1,2 -> 0,6388 / 0,6382 (contre 0,6365 / 0,6350), 1,25 -> 0,6390 / 0,6383, 1,3 -> 0,6374 / 0,6373 ; poids hard 0,93 ou 1,07 neutre : plateau, on garde 1,2 | oui (soumission 17:52) |
| 17:48 | Équilibrage par (personne, tranche de temps), cibles rétrécies (driver 2) | la forme évolue : début d'historique plus dur | +0,003 | 0,6293 / 0,6267 (2 tranches), 0,6251 / 0,6201 (3) : parts par tranche trop bruitées | non |
| 17:50 | Rang temporel dans un logreg de 2e niveau + équilibrage (driver 2) | capter la dérive temporelle | +0,002 | 0,6343 / 0,6318 (sans rang 0,6354 / 0,6325) | non |
| 17:51 | Équilibrer plusieurs jeux de probas puis voter (driver 2) | diversité des décisions | +0,001 | 0,6361 / 0,6345 contre 0,6358 / 0,6332 : bruit | non |
| 17:49 | Pseudo-étiquetage itératif des séances sûres après équilibrage (driver 2) | affiner les comptes des séances restantes | +0,001 | meilleur 0,6359 / 0,6350 (seuil 0,8, 2 tours) : sous le seuil | non |
| 17:53 | Méta-modèle enrichi (driver 4) : méta-features (sport, nombre de séances, rang temporel), rangs intra-personne des probas, méta-learner HGB, log-probas centrées par personne | rapprocher la calibration par personne dès le stacking | +0,002 | méta-features 0,6263, rangs 0,6255, HGB 0,6210 (contre 0,6271) ; log-probas centrées : 0,6284, après équilibrage 0,6367 / 0,6347 (+0,0009 / +0,0015, sous le seuil) | non |

## Meilleure idée (pour le jury)

____
