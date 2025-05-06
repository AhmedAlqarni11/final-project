import openai
from openai import OpenAI

# Set OpenAI API Key
OPENAI_API_KEY = "your-api-key"
openai.api_key = OPENAI_API_KEY

client = OpenAI()

def summarize_text(text):
    """Summarize text using OpenAI GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are a helpful assistant."
            },
            {
                "role": "user", "content": f"Summarize the following text briefly:\n\n{text}"
            }],
            temperature=0.4,
            max_tokens=200
        )

        summary_text = response.choices[0].message.content.strip()
        return summary_text

    except Exception as e:
        return f"❌ Error summarizing text: {e}"

def translate_text(text, target_language="en"):
    """Translate text using OpenAI GPT"""
    try:
        # Translate the text using OpenAI GPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system", "content": "You are a helpful assistant that translates text."
            },
            {
                "role": "user", "content": f"Translate the following text to {target_language}:\n\n{text}"
            }],
            temperature=0.4,
            max_tokens=200
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text  # Return translated text

    except Exception as e:
        return f"❌ Error during text translation: {e}"
