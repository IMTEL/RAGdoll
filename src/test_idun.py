import sys
import requests
 
def chat_with_model(token,model,question):
    url = 'https://idun-llm.hpc.ntnu.no/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": question
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()
 
my_api_key = "sk-dbc483d4e3c3479e9be50e91d190d5d9"
my_model = "openai/gpt-oss-120b"
my_question = "tell me about flamingos"
answer = chat_with_model(my_api_key, my_model, my_question)
print(answer)