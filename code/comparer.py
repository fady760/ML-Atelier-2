import json
from pathlib import Path

DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"


def charger_resultats():
    fichiers = list(DOSSIER_RESULTATS.glob("*.json"))
    resultats = []
    for f in sorted(fichiers):
        with open(f, encoding="utf-8") as flux:
            resultats.append(json.load(flux))
    return resultats


def afficher_tableau_comparatif(resultats):
    print("\nCOMPARAISON DES MODÈLES")
    print("-" * 70)
    print(f"{'Modèle':<34} {'Accuracy':>8} {'Macro-F1':>9} {'Invalides':>10} {'Durée (s)':>10}")
    for r in resultats:
        nom = r.get("etiquette", r.get("modele", "inconnu"))
        acc = r.get("accuracy", 0.0)
        mf1 = r.get("macro_f1", 0.0)
        inv = r.get("invalides", 0.0)
        duree = r.get("duree_s", 0.0)
        print(f"{nom:<34} {acc:>8.3f} {mf1:>9.3f} {inv:>10.3f} {duree:>10.1f}")
    print()


def afficher_desaccords(resultats):
    if len(resultats) < 2:
        return

    print("DÉSACCORDS ENTRE MODÈLES")
    ids = resultats[0].get("ids", [])
    nb_tickets = len(ids)

    for i in range(nb_tickets):
        ticket_id = ids[i]
        predictions = [r["predits"][i] for r in resultats]

        # S'il y a au moins deux prédictions différentes
        if len(set(predictions)) > 1:
            details = " | ".join(
                f"{r.get('etiquette', r.get('modele'))} = {r['predits'][i]}"
                for r in resultats
            )
            print(f"{ticket_id} : {details}")
    print()


def main():
    resultats = charger_resultats()
    if not resultats:
        print(f"Aucun résultat trouvé dans : {DOSSIER_RESULTATS}")
        return

    afficher_tableau_comparatif(resultats)
    afficher_desaccords(resultats)


if __name__ == "__main__":
    main()
