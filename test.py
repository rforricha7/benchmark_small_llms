import requests
import json
import urllib3

# suppress LibreSSL warning
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

url = "http://localhost:11434/v1/generate"

payload = {
    "model": "llama3:8b",
    "prompt": "Hello Ollama!",
    "stream": False
}

r = requests.post(url, json=payload, verify=False)

# Ollama may return multiple JSON lines; take the first line with "response"
lines = r.text.splitlines()
output = None

for line in lines:
    try:
        data = json.loads(line)
        if "response" in data:
            output = data["response"].strip()
            break
    except json.JSONDecodeError:
        continue  # skip lines that aren’t valid JSON

if output:
    print("Model output:", output)
else:
    print("No valid response found. Raw output:")
    print(r.text)

