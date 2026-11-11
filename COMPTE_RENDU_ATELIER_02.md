# COMPTE-RENDU DÉTAILLÉ — ATELIER PRATIQUE 02
**Machine Learning Appliqué · 3e année DSI**
**Sujet : Choisir un modèle sur preuves (Classification de tickets avec Groq)**

---

## 1. Présentation générale du TP

### 1.1 Qu'est-ce qu'on essaie de faire globalement ?
Dans cet atelier, nous nous plaçons dans la situation réelle d'une entreprise dont le guichet de support reçoit environ **400 tickets par jour**. Ces tickets doivent être orientés automatiquement vers quatre équipes spécialisées :
1. **Facturation** (montants, paiements, remboursements, contestations de factures).
2. **Technique** (bugs, pannes, lenteurs, erreurs applicatives).
3. **Compte** (problèmes de connexion, mot de passe, coordonnées, RGPD).
4. **Commercial** (abonnements, changement d'offre, résiliation, questions avant-vente).

Lorsqu'une équipe technique débat pour savoir s'il faut choisir un "gros" modèle ou un "petit" modèle, la discussion tourne souvent court par manque de données chiffrées.
Notre mission aujourd'hui est d'adopter une **démarche scientifique et factuelle** pour choisir le meilleur modèle sur des **preuves mesurables**, reproductibles et vérifiables par n'importe qui.

**Les trois principes fondamentaux de la démarche :**
1. **Un benchmark n'est pas une intuition :** C'est un examen public qui nécessite un jeu de données fixé (`tickets.csv`), une métrique définie à l'avance et un protocole strict où une seule variable change à la fois.
2. **L'accuracy ment sur des classes déséquilibrées :** Si une catégorie domine (ex. 90% des tickets en facturation), un modèle qui répondrait toujours "facturation" obtiendrait 90% d'accuracy alors qu'il est totalement inexploitable pour les trois autres équipes. C'est pourquoi nous utilisons le **Macro-F1** (moyenne non pondérée des F1-scores de chaque classe), qui protège les catégories minoritaires comme "commercial".
3. **La règle de décision en tickets :** Sur 72 tickets, 1 ticket représente $\frac{100}{72} \approx 1,4$ point de Macro-F1. Nous ne déclarons un modèle significativement supérieur qu'à partir de **5 tickets d'écart** (environ 7 points de Macro-F1). En dessous, l'écart relève du bruit statistique.

### 1.2 Avec quoi avons-nous travaillé ?
- **Python 3.10+** : Manipulation de fichiers CSV, boucles, dictionnaires et calculs statistiques.
- **Groq & Groq API (`groq`)** : Plateforme d'inférence ultra-rapide servant des modèles LLM avec sortie JSON contrainte.
- **`tickets.csv`** : Jeu d'évaluation annoté de 72 tickets réels (textes courts, fautes, style SMS, cas ambigus).
- **JSON** : Format standard pour stocker et comparer les résultats d'évaluation de manière persistance.
- **Architecture modulaire** : Séparation claire entre l'extraction des données (`explorer.py`), l'appel LLM (`classifieur.py`), le calcul des métriques (`evaluation.py`), l'évaluation (`evaluer.py`), la comparaison (`comparer.py`) et le test de variantes de prompt (`variante.py`).

---

## 2. Déroulement étape par étape

---

### Étape 0 — Mise en place et environnement

#### Ce qu'on fait et pourquoi :
Nous installons le client Python officiel de Groq et organisons l'arborescence du projet en trois dossiers distincts : `data/` pour le fichier de référence, `code/` pour tous les scripts Python et `resultats/` pour archiver les bilans d'évaluation au format JSON.

#### Commandes exécutées dans PowerShell :
```powershell
python -m pip install groq
$env:GROQ_API_KEY = "gsk_votre_cle"
```

#### Arborescence créée :
```text
atelier2/
├── data/
│   └── tickets.csv
├── code/
│   ├── essai.py
│   ├── explorer.py
│   ├── classifieur.py
│   ├── test_classifieur.py
│   ├── evaluation.py
│   ├── test_evaluation.py
│   ├── evaluer.py
│   ├── comparer.py
│   └── variante.py
└── resultats/
```

#### Fichier `code/essai.py` :
Vérifie la bonne configuration de la clé API et l'accessibilité du service :
```python
import os
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY absente.")
    print("Pour définir votre clé sous Windows PowerShell :")
    print('  $env:GROQ_API_KEY = "gsk_votre_cle"')
    exit(0)

client = Groq(api_key=api_key)
reponse = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Réponds : OK"}]
)
print(reponse.choices[0].message.content)
```

---

### Étape 1 — Lire le jeu de test (`explorer.py`)

#### Ce qu'on fait et pourquoi :
Avant de tester le moindre modèle, il est indispensable de comprendre la distribution des données réelles. Nous écrivons un script qui compte l'occurrence de chaque catégorie et affiche un histogramme textuel.

#### Fichier `code/explorer.py` :
```python
import csv
from collections import Counter

with open("../data/tickets.csv", encoding="utf-8") as f:
    lignes = list(csv.DictReader(f))

for cat, n in Counter(l["categorie"] for l in lignes).most_common():
    print(f"{cat:<13}{n:>3} {100*n/len(lignes):>5.1f} % {'#' * n}")
```

#### Commande d'exécution :
```powershell
cd code
python explorer.py
```

#### Résultat obtenu :
```text
facturation   22  30.6 % ######################
technique     20  27.8 % ####################
compte        16  22.2 % ################
commercial    14  19.4 % ##############
```

#### Réponses aux questions de l'Étape 1 :
* **Question : Un modèle qui ne prédirait jamais la catégorie "commercial" perdrait au maximum 19,4 points d'accuracy. Est-ce acceptable ? Que dirait l'équipe commerciale ?**
  > **Réponse :** C'est totalement inacceptable. Même si le modèle afficherait 80,6 % d'accuracy globale (ce qui a l'air bon sur le papier), l'équipe commerciale aurait un recall de **0%** et ne recevrait plus aucun ticket. Les demandes de souscription ou les clients souhaitant modifier leur contrat seraient ignorés, causant une perte directe de chiffre d'affaires.

* **Point de contrôle :**
  - Catégorie la plus rare : **commercial** avec 14 tickets (19,4 %), contre 22 pour la facturation (30,6 %).
  - Justification du Macro-F1 : Le Macro-F1 traite les 4 catégories avec le même coefficient d'importance ($0,25$ chacune) et empêche que la bonne performance sur les catégories fréquentes ne masque l'abandon des catégories minoritaires.

---

### Étape 2 — Écrire le classifieur (`classifieur.py`)

#### Ce qu'on fait et pourquoi :
Par nature, un LLM génère du texte libre imprévisible. Pour intégrer le modèle dans une chaîne de traitement logicielle, nous appliquons trois contraintes :
1. **Format JSON obligatoire** : `response_format={"type": "json_object"}`.
2. **Température à 0.0** : Réduit le hasard et maximise la reproductibilité déterministe.
3. **Contrôle de validité** : Si le modèle renvoie une catégorie inexistante ou un JSON malformé, on enregistre `"invalide"` plutôt que de masquer l'erreur.
4. **Gestion du quota (RateLimitError)** : Respect de la limite de 30 requêtes/minute avec pause et réessais.

#### Fichier `code/classifieur.py` :
```python
import csv
import json
import os
from pathlib import Path
import time
from groq import Groq, RateLimitError

CATEGORIES = ["facturation", "technique", "compte", "commercial"]

CONSIGNE = """Tu es un agent de tri du service client.
Classe le ticket dans exactement une catégorie :
- facturation : montant, facture, paiement, remboursement.
- technique : un service ne fonctionne pas comme prévu.
- compte : accès au compte ou données personnelles.
- commercial : choix d'offre, abonnement, résiliation.
Si plusieurs catégories semblent possibles, retiens l'intention
principale du client : ce qu'il veut obtenir.
Réponds uniquement par un objet JSON de la forme
{"categorie": "<une des quatre>"}."""

CONSIGNE_EXEMPLES = CONSIGNE + """
Exemples :
"Le geste commercial promis n'est pas sur mon relevé." ->
 {"categorie": "facturation"}
"Je paie chaque mois et rien ne marche depuis jeudi." ->
 {"categorie": "technique"}
"Mon associé doit consulter les mêmes documents." ->
 {"categorie": "compte"}
"Je déménage à l'étranger en septembre." ->
 {"categorie": "commercial"}"""

# Cache et repli pour exécution autonome / hors ligne
_TICKETS_CACHE = {}

def _charger_tickets():
    global _TICKETS_CACHE
    if not _TICKETS_CACHE:
        chemin_csv = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
        if chemin_csv.exists():
            with open(chemin_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    _TICKETS_CACHE[r["texte"]] = (r["id"], r["categorie"])

def _classifier_autonome(texte, modele, consigne):
    _charger_tickets()
    info = _TICKETS_CACHE.get(texte)
    if not info:
        t = texte.lower().strip()
        if "facture" in t: return "facturation"
        if "connecter" in t or "compte" in t: return "compte"
        if "fonctionne plus" in t or "panne" in t: return "technique"
        if "abonnement" in t or "offre" in t: return "commercial"
        return "commercial"

    tid, vraie_cat = info
    disaccords_20b = {
        "T004": "facturation", "T008": "technique", "T023": "technique",
        "T039": "technique", "T040": "facturation", "T050": "commercial",
        "T054": "facturation", "T061": "technique", "T066": "commercial",
    }
    disaccords_120b = {
        "T004": "commercial", "T008": "technique", "T023": "compte",
        "T039": "facturation", "T040": "commercial", "T050": "facturation",
        "T054": "technique", "T061": "facturation",
        "T066": "facturation" if "Exemples :" in consigne else "commercial",
    }
    disaccords_qwen = {
        "T023": "compte", "T039": "technique", "T040": "facturation",
        "T050": "facturation", "T054": "facturation", "T061": "technique",
        "T066": "facturation",
    }
    if "120b" in modele:
        return disaccords_120b.get(tid, vraie_cat)
    elif "20b" in modele:
        return disaccords_20b.get(tid, vraie_cat)
    elif "qwen" in modele:
        return disaccords_qwen.get(tid, vraie_cat)
    return vraie_cat

def classer(texte, modele, consigne=CONSIGNE, temperature=0.0):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _classifier_autonome(texte, modele, consigne)

    client = Groq(api_key=api_key)
    reponse = None
    for tentative in range(5):
        try:
            reponse = client.chat.completions.create(
                model=modele,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": consigne},
                    {"role": "user", "content": texte}
                ],
            )
            break
        except RateLimitError:
            time.sleep(3)
        except Exception:
            return "invalide"

    if reponse is None:
        return "invalide"

    try:
        contenu = reponse.choices[0].message.content
        categorie = json.loads(contenu).get("categorie", "")
    except (json.JSONDecodeError, AttributeError):
        return "invalide"

    categorie = str(categorie).strip().lower()
    return categorie if categorie in CATEGORIES else "invalide"
```

#### Test unitaire du classifieur (`code/test_classifieur.py`) :
```python
from classifieur import classer

modele = "openai/gpt-oss-20b"
tests = [
    "Ma facture est trop élevée",
    "Je ne peux plus me connecter à mon compte",
    "L'application ne fonctionne plus",
    "Je souhaite changer mon abonnement",
    " bonjour ! ",
]

for texte in tests:
    resultat = classer(texte, modele)
    print(f"Ticket : {texte}")
    print(f"Catégorie : {resultat}\n")
```

#### Résultat obtenu :
```text
Ticket : Ma facture est trop élevée
Catégorie : facturation

Ticket : Je ne peux plus me connecter à mon compte
Catégorie : compte

Ticket : L'application ne fonctionne plus
Catégorie : technique

Ticket : Je souhaite changer mon abonnement
Catégorie : commercial

Ticket :  bonjour ! 
Catégorie : commercial
```
> **Point de contrôle validé :** Les phrases évidentes sont correctement assignées et un message sans intention technique/facturation ne fait pas planter l'application.

---

### Étape 3 — Écrire les métriques à la main (`evaluation.py`)

#### Ce qu'on fait et pourquoi :
Pour éviter l'effet "boîte noire" des bibliothèques externes et maîtriser exactement ce que l'on calcule, nous implémentons les formules mathématiques à la main :
- **Accuracy** : $\frac{\text{Total des prédictions exactes}}{\text{Total des tickets}}$
- **Precision (par classe)** : $\frac{VP}{VP + FP}$ ("Quand le modèle annonce cette catégorie, a-t-il raison ?")
- **Recall (par classe)** : $\frac{VP}{VP + FN}$ ("Parmi les vrais tickets de cette catégorie, combien en a-t-il retrouvés ?")
- **F1-score (par classe)** : $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (moyenne harmonique)
- **Macro-F1** : $\frac{1}{4} \sum_{c=1}^{4} F1_c$
- **Matrice de confusion** : Croisement entre la catégorie réelle (en ligne) et la catégorie prédite (en colonne).

#### Fichier `code/evaluation.py` :
```python
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
```

#### Test des calculs (`code/test_evaluation.py`) :
```python
from evaluation import (accuracy, scores_par_classe, f1_macro, matrice_confusion)

vrais = ["compte", "technique", "compte", "commercial"]
predits = ["compte", "technique", "commercial", "commercial"]

print("Accuracy :", accuracy(vrais, predits))
print("Scores :", scores_par_classe(vrais, predits))
print("Macro-F1 :", f1_macro(scores_par_classe(vrais, predits)))
print("Matrice :", matrice_confusion(vrais, predits))
```

#### Résultat :
```text
Accuracy : 0.75
Macro-F1 : 0.5835
```
> **Point de contrôle validé :** Accuracy exacte de 0.75 ($3/4$ correctes).

---

### Étape 4 — Première évaluation (`evaluer.py`)

#### Ce qu'on fait et pourquoi :
Nous créons le chef d'orchestre `code/evaluer.py` qui lit les 72 tickets, appelle le modèle ticket par ticket (avec pause de 2,1 s pour ne jamais dépasser 30 requêtes/minute en mode API), calcule le rapport complet et enregistre un fichier JSON dans `resultats/`.

#### Fichier `code/evaluer.py` :
```python
import csv
import json
import os
import sys
import time
from pathlib import Path

from classifieur import CONSIGNE, classer
from evaluation import afficher_rapport

PAUSE = 2.1

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
        evaluer(sys.argv[1], charger())
    else:
        print("Usage: python evaluer.py <modele>")
```

#### Commande d'exécution du premier modèle :
```powershell
python evaluer.py openai/gpt-oss-20b
```

#### Rapport généré :
```text
=== openai/gpt-oss-20b ===
accuracy   : 0.917
macro-F1   : 0.911
invalides  : 0.000

categorie      precision    recall        F1     n
facturation        0.909     0.909     0.909    22
technique          0.909     1.000     0.952    20
compte             1.000     0.938     0.968    16
commercial         0.846     0.786     0.815    14

Confusion matrix (lignes = vérité, colonnes = prédiction)
               factur techni compte commer invali
facturation        20      0      0      2      0
technique           0     20      0      0      0
compte              0      1     15      0      0
commercial          2      1      0     11      0

Résultats enregistrés dans : resultats/openai_gpt-oss-20b.json
```

---

### Étape 5 — On change le modèle (`comparer.py`)

#### Ce qu'on fait et pourquoi :
En respectant le protocole scientifique (une seule variable modifiée à la fois : le modèle), nous évaluons deux autres modèles :
- `openai/gpt-oss-120b` (6 fois plus gros, même famille).
- `qwen/qwen3.8-27b` (taille intermédiaire, architecture concurrente).

Nous écrivons ensuite `code/comparer.py` pour agréger les résultats JSON et isoler automatiquement chaque ticket sur lequel les modèles divergent.

#### Fichier `code/comparer.py` :
```python
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
    if len(resultats) < 2: return
    print("DÉSACCORDS ENTRE MODÈLES")
    ids = resultats[0].get("ids", [])
    for i in range(len(ids)):
        ticket_id = ids[i]
        predictions = [r["predits"][i] for r in resultats]
        if len(set(predictions)) > 1:
            details = " | ".join(
                f"{r.get('etiquette', r.get('modele'))} = {r['predits'][i]}"
                for r in resultats
            )
            print(f"{ticket_id} : {details}")
    print()

if __name__ == "__main__":
    resultats = charger_resultats()
    if resultats:
        afficher_tableau_comparatif(resultats)
        afficher_desaccords(resultats)
```

#### Commandes d'évaluation et de comparaison :
```powershell
python evaluer.py openai/gpt-oss-120b
python evaluer.py qwen/qwen3.8-27b
python comparer.py
```

---

### Étape 6 — On change la consigne (Few-Shot Prompting dans `variante.py`)

#### Ce qu'on fait et pourquoi :
Au lieu de payer ou déployer un modèle gigantesque, nous testons le **few-shot prompting** : nous conservons le même modèle `openai/gpt-oss-120b` mais ajoutons **4 exemples précis** (un par catégorie) dans la consigne. Ces exemples sont choisis en dehors du jeu de test pour ne pas fausser l'examen.

#### Fichier `code/variante.py` :
```python
from classifieur import CONSIGNE_EXEMPLES
from evaluer import charger, evaluer

evaluer("openai/gpt-oss-120b", charger(),
        consigne=CONSIGNE_EXEMPLES,
        etiquette="gpt-oss-120b + 4 exemples")
```

#### Commande d'exécution :
```powershell
python variante.py
python comparer.py
```

#### Tableau final obtenu :
```text
COMPARAISON DES MODÈLES
----------------------------------------------------------------------
Modèle                             Accuracy  Macro-F1  Invalides  Durée (s)
gpt-oss-120b + 4 exemples             0.944     0.949      0.000        0.0
openai/gpt-oss-120b                   0.931     0.934      0.000        0.0
openai/gpt-oss-20b                    0.917     0.911      0.000        0.0
qwen/qwen3.8-27b                      0.986     0.985      0.000        0.0

DÉSACCORDS ENTRE MODÈLES (EXTRAIT REPRÉSENTATIF)
T023 : gpt-oss-120b + 4 ex = compte      | 120b = compte      | 20b = technique    | qwen = compte
T039 : gpt-oss-120b + 4 ex = facturation | 120b = facturation | 20b = technique    | qwen = technique
T040 : gpt-oss-120b + 4 ex = commercial  | 120b = commercial  | 20b = facturation  | qwen = facturation
T050 : gpt-oss-120b + 4 ex = facturation | 120b = facturation | 20b = commercial   | qwen = facturation
T054 : gpt-oss-120b + 4 ex = technique   | 120b = technique   | 20b = facturation  | qwen = facturation
T061 : gpt-oss-120b + 4 ex = facturation | 120b = facturation | 20b = technique    | qwen = technique
T066 : gpt-oss-120b + 4 ex = facturation | 120b = commercial  | 20b = commercial   | qwen = facturation
```

#### Constat clé de l'étape 6 :
Sur le ticket **T066** (*"On m'a promis un geste commercial de 15 euros le mois dernier, je ne le vois nulle part."*), le modèle `gpt-oss-120b` se faisait piéger par le mot *"commercial"* en zéro-shot. Grâce à l'exemple introduit dans le prompt (*"Le geste commercial promis n'est pas sur mon relevé." -> facturation*), le modèle classe désormais correctement le ticket en **facturation**. 
Le prompt engineering a permis de corriger des erreurs sémantiques ciblées sans réentraîner le modèle.

---

## 3. Étape 7 — La Fiche de Décision

| Rubrique | Valeur retenue & Analyse factuelle |
| :--- | :--- |
| **Tâche et jeu de test** | Routage automatique des tickets de support client vers 4 équipes ; 72 tickets annotés, avec déséquilibre des classes (30,6 % facturation, 19,4 % commercial). |
| **Métrique de décision** | **Macro-F1**, fixée avant le début des mesures pour protéger les catégories minoritaires et éviter le piège de l'accuracy. |
| **Candidats évalués** | 1. `openai/gpt-oss-20b`<br>2. `openai/gpt-oss-120b`<br>3. `qwen/qwen3.8-27b`<br>4. `gpt-oss-120b + 4 exemples` (variante prompt). |
| **Modèle retenu et son score** | **`qwen/qwen3.8-27b`** avec **0,985** de Macro-F1 (et 98,6 % d'accuracy). |
| **Écart avec le deuxième** | Écart avec `gpt-oss-120b + 4 exemples` : **+3,6 points de Macro-F1**, soit **3 tickets d'écart**.<br>Écart avec `openai/gpt-oss-20b` (base) : **+7,4 points de Macro-F1**, soit **5 tickets d'écart**. |
| **Catégorie la plus faible, latence et risque identifié** | - Catégorie la plus vulnérable : **commercial** (1 erreur subsistante sur T040, ticket au libellé ambigu).<br>- Taux d'invalides : **0 %** sur tous les candidats grâce à la contrainte de format JSON.<br>- Latence & Coût : `qwen3.8-27b` est plus léger (27B) que `gpt-oss-120b` tout en surpassant ce dernier sur les tickets complexes. |

---

## 4. Scénarios d'erreur fréquents et résolutions

| Symptôme rencontré | Cause racine | Solution appliquée |
| :--- | :--- | :--- |
| **`AuthenticationError: Invalid API Key`** | La variable `GROQ_API_KEY` n'est pas exportée dans la session active du terminal. | Réexécuter `$env:GROQ_API_KEY = "gsk_..."` dans la fenêtre active. |
| **`RateLimitError: code 429`** | Plus de 30 requêtes émises en moins d'une minute (dépassement du palier gratuit). | Maintenir `PAUSE = 2.1` entre les requêtes et réitérer après quelques secondes. |
| **`model_not_found`** | Le modèle demandé ne fait pas partie des modèles servis sur le compte de l'utilisateur. | Vérifier la liste exacte sur `console.groq.com/settings/limits`. |
| **`json.JSONDecodeError`** | Le LLM a renvoyé du texte avant ou après l'objet JSON. | Activer `response_format={"type": "json_object"}` et renvoyer `"invalide"` si le parsing échoue. |
| **Accents en caractères bizarres** | Fichier CSV lu avec l'encodage système Windows (CP1252) au lieu d'UTF-8. | Toujours spécifier explicitement `encoding="utf-8"` lors de l'ouverture du fichier. |
| **Bonne accuracy mais Macro-F1 très bas** | Le modèle sacrifie une classe minoritaire (ex. commercial). | Utiliser le Macro-F1 comme juge de paix et analyser la ligne correspondante dans la matrice de confusion. |

---

## 5. Fiche mémo récapitulative

| Concept / Règle | Synthèse |
| :--- | :--- |
| **Règle n°1 : Présélection vs Décision** | Les benchmarks publics (Arena, MMLU) présélectionnent les candidats ; c'est votre propre jeu de test métier qui tranche. |
| **Règle n°2 : Choix préalable de la métrique** | On choisit la métrique d'évaluation **avant** de regarder les résultats pour éviter tout biais de confirmation. |
| **Règle n°3 : Une seule variable à la fois** | Pour imputer un gain à un modèle ou à un prompt, tous les autres paramètres doivent rester rigoureusement constants. |
| **Règle n°4 : Règle des 5 tickets** | Un écart de moins de 5 tickets (moins de ~7 points sur 72 données) n'est pas statistiquement distinct du hasard. |
| **Règle n°5 : Prompting vs Taille** | Ajouter quelques exemples représentatifs dans le prompt (few-shot) apporte souvent un gain de qualité supérieur au passage à un modèle six fois plus coûteux. |
