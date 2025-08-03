import requests

def Ask(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": "<API_KEY>"  # Replace with your actual API key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Response format unexpected."
    else:
        return f"Request failed with status code {response.status_code}: {response.text}"


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = Ask(prompt)
    print(response)