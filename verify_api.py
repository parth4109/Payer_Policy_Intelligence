import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from groq import Groq


key = os.getenv("GROQ_API_KEY")



client = Groq(api_key=key)

try:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Reply with only the word: OK"}],
        temperature=0,
        max_tokens=5,
    )
    print("API KEY VALID [OK]")
    print("Model response:", resp.choices[0].message.content.strip())
    print("Model used:", resp.model)
    print("Request ID:", resp.id)
except Exception as e:
    print(f"API KEY ERROR: {e}")
