from classifieur import CONSIGNE_EXEMPLES
from evaluer import charger, evaluer

evaluer("openai/gpt-oss-120b", charger(),
        consigne=CONSIGNE_EXEMPLES,
        etiquette="gpt-oss-120b + 4 exemples")
