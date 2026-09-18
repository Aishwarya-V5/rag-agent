from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
c = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))

for model in ['open-mistral-nemo', 'ministral-8b-2512', 'mistral-large-latest', 'open-mistral-7b']:
    try:
        r = c.chat.complete(model=model, messages=[{'role': 'user', 'content': 'hi'}])
        print(f'{model}: SUCCESS')
    except Exception as e:
        print(f'{model}: FAILED - {e}')