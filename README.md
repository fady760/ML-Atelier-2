# Atelier 2 — Choisir un modèle sur preuves (Classification de tickets)

Machine Learning Appliqué · 3e année DSI

## Objectif
Automatiser le routage de 400 tickets/jour vers quatre équipes (`facturation`, `technique`, `compte`, `commercial`) en évaluant scientifiquement plusieurs modèles d'IA sur un jeu de test de 72 tickets annotés.

## Structure du projet
- `data/` : Contient `tickets.csv` (72 tickets réels annotés).
- `code/` : Scripts de tri, d'évaluation, de calcul de métriques et de comparaison.
- `resultats/` : Résultats des évaluations au format JSON.
- `COMPTE_RENDU_ATELIER_02.md` : Compte-rendu complet étape par étape et fiche de décision finale.

## Utilisation

```powershell
cd code

# 1. Explorer la répartition des tickets
python explorer.py

# 2. Tester le classifieur
python test_classifieur.py

# 3. Évaluer un modèle (ex: openai/gpt-oss-20b)
python evaluer.py openai/gpt-oss-20b

# 4. Évaluer la variante de prompt (few-shot avec 4 exemples)
python variante.py

# 5. Comparer tous les modèles et voir les désaccords
python comparer.py
```
