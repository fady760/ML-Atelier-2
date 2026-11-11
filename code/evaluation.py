CATEGORIES = ["facturation", "technique", "compte", "commercial"]


def accuracy(vrais, predits):
    return sum(1 for v, p in zip(vrais, predits) if v == p) / len(vrais)


def scores_par_classe(vrais, predits, classes=CATEGORIES):
    resultat = {}
    for c in classes:
        vp = sum(1 for v, p in zip(vrais, predits) if v == c and p == c)
        fp = sum(1 for v, p in zip(vrais, predits) if v != c and p == c)
        fn = sum(1 for v, p in zip(vrais, predits) if v == c and p != c)
        precision = vp / (vp + fp) if (vp + fp) else 0.0
        recall = vp / (vp + fn) if (vp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        resultat[c] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": vp + fn
        }
    return resultat


def f1_macro(scores):
    return sum(d["f1"] for d in scores.values()) / len(scores)


def taux_invalides(predits):
    return sum(1 for p in predits if p == "invalide") / len(predits) if predits else 0.0


def matrice_confusion(vrais, predits, classes=CATEGORIES):
    colonnes = classes + ["invalide"]
    matrice = {c_vrai: {c_pred: 0 for c_pred in colonnes} for c_vrai in classes}
    for v, p in zip(vrais, predits):
        if v in matrice:
            if p in colonnes:
                matrice[v][p] += 1
            else:
                matrice[v]["invalide"] += 1
    return matrice


def afficher_rapport(etiquette, vrais, predits):
    acc = accuracy(vrais, predits)
    scores = scores_par_classe(vrais, predits)
    mf1 = f1_macro(scores)
    inv = taux_invalides(predits)
    mat = matrice_confusion(vrais, predits)

    print(f"\n=== {etiquette} ===")
    print(f"accuracy   : {acc:.3f}")
    print(f"macro-F1   : {mf1:.3f}")
    print(f"invalides  : {inv:.3f}\n")

    print(f"{'categorie':<14} {'precision':>9} {'recall':>9} {'F1':>9} {'n':>5}")
    for cat, sc in scores.items():
        print(f"{cat:<14} {sc['precision']:>9.3f} {sc['recall']:>9.3f} {sc['f1']:>9.3f} {sc['support']:>5}")

    print("\nConfusion matrix (lignes = vérité, colonnes = prédiction)")
    print(f"{'':<14} {'factur':>6} {'techni':>6} {'compte':>6} {'commer':>6} {'invali':>6}")
    for cat in CATEGORIES:
        print(f"{cat:<14} {mat[cat]['facturation']:>6} {mat[cat]['technique']:>6} {mat[cat]['compte']:>6} {mat[cat]['commercial']:>6} {mat[cat]['invalide']:>6}")
    print()

    return {
        "accuracy": round(acc, 3),
        "macro_f1": round(mf1, 3),
        "invalides": round(inv, 3),
        "scores_par_classe": scores,
        "matrice_confusion": mat
    }
