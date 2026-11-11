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
