#!/usr/bin/env python3
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set your API key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

def chat():
    print("GPT4O API Chat is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-0613",  # Replace with your preferred model if necessary
                messages=[{"role": "user", "content": user_input}],
                max_tokens=150,
                temperature=0.7,
            )
            message = response["choices"][0]["message"]["content"].strip()
            print("AI:", message, "\n")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    chat()
