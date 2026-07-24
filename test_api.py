from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

print(os.getenv("OPENROUTER_API_KEY"))


response = client.chat.completions.create(
    model="inclusionai/ling-3.0-flash:free",
    messages=[
        {
            "role": "user",
            "content": "Explain Artificial Intelligence in simple words"
        }
    ]
)

print(response.choices[0].message.content)