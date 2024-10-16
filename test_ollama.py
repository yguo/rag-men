import requests
import json

# def generate_with_ollama(prompt, model="llama2"):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": model,
#         "prompt": prompt
#     }
#     try:
#         with requests.post(url, json=payload, stream=True) as response:
#             response.raise_for_status()
#             for line in response.iter_lines():
#                 if line:
#                     try:
#                         yield json.loads(line)
#                     except json.JSONDecodeError as json_error:
#                         print(f"Error decoding JSON: {json_error}")
#                         print(f"Problematic line: {line}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error: {e}")
#         if hasattr(e, 'response'):
#             print(f"Status code: {e.response.status_code}")
#             print(f"Response content: {e.response.text}")

# # Test the function
# full_response = ""
# for chunk in generate_with_ollama("Hello, world!"):
#     if 'response' in chunk:
#         full_response += chunk['response']
#     print(json.dumps(chunk, indent=2))

# print("\nFull response:")
# print(full_response)


# import ollama
# response = ollama.chat(model='llama3.1', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])



import requests
import json

url = "http://localhost:11434/api/chat"
data = {
    "model": "llama2",
    "messages": [
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        }
    ],
    "stream": False
}

response = requests.post(url, json=data)

if response.status_code == 200:
    response_data = response.json()
    print(json.dumps(response_data, indent=4))
else:
    print(f"status_code{response.status_code}")
    print(response.text)