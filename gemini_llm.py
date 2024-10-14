
import requests
import json

class GeminiLLM:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(f"{self.api_url}?key={self.api_key}", headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                 try:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
                 except (KeyError, IndexError) as e:
                    print(f"Failed to extract text from response: {e}")
                    return "Error: No response generated"
                            
            else:
                return "The model did not return a valid test."
        else:
            return f"Error: {response.status_code} - {response.text}"






