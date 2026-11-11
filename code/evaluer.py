import csv
import json
import os
import sys
import time
from pathlib import Path

from classifieur import CONSIGNE, classer
from evaluation import afficher_rapport

PAUSE = 2.1  # 30 requêtes/minute au maximum


def charger(chemin="../data/tickets.csv"):
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluer(modele, lignes, consigne=CONSIGNE, etiquette=None):
    etiquette = etiquette or modele
    vrais, predits, debut = [], [], time.time()

    print(f"\nÉvaluation en cours pour : {etiquette} ({len(lignes)} tickets)...")
    for ligne in lignes:
        pred = classer(ligne["texte"], modele, consigne=consigne)
        predits.append(pred)
        vrais.append(ligne["categorie"])
        if os.environ.get("GROQ_API_KEY"):
            time.sleep(PAUSE)

    resume = afficher_rapport(etiquette, vrais, predits)
    resume.update(
        modele=modele,
        etiquette=etiquette,
        duree_s=round(time.time() - debut, 1),
        predits=predits,
        ids=[l["id"] for l in lignes]
    )

    # Dossier resultats existant
    DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"
    DOSSIER_RESULTATS.mkdir(exist_ok=True)

    nom_fichier = etiquette.replace("/", "_") + ".json"
    fichier_sortie = DOSSIER_RESULTATS / nom_fichier

    with open(fichier_sortie, "w", encoding="utf-8") as f:
        json.dump(resume, f, ensure_ascii=False, indent=2)

    print(f"Résultats enregistrés dans : {fichier_sortie}")
    return resume


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modele_choisi = sys.argv[1]
        evaluer(modele_choisi, charger())
    else:
        print("Usage: python evaluer.py <modele>")
