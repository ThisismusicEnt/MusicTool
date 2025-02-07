#!/usr/bin/env python3
"""
chat.py
-------
A minimal chat interface for TinyLLaMA.
Make sure the TinyLLaMA repository is cloned and its files are in the proper location.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./TinyLLaMA"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading TinyLLaMA: {e}")
        return None, None

def chat():
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        print("Failed to load TinyLLaMA model.")
        return

    print("TinyLLaMA loaded successfully. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("AI:", response, "\n")

if __name__ == "__main__":
    chat()
