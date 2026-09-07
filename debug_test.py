from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
c = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))

for model in ['ministral-8b-2512', 'mistral-medium-latest', 'open-mistral-nemo']:
    try:
        r = c.chat.complete(model=model, messages=[{'role': 'user', 'content': 'hi'}])
        print(f"{model}: SUCCESS - {r.choices[0].message.content}")
    except Exception as e:
        print(f"{model}: FAILED - {e}")