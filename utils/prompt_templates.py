# utils/prompt_templates.py

def apply_persona(prompt, style="default"):
    personas = {
        "default": lambda x: x,
        "shakespeare": lambda x: f"Speak as if in days of old: {x}",
        "tech_support": lambda x: f"You are a helpful tech support assistant. User asks: {x}",
        "child_friendly": lambda x: f"Explain this like I’m five: {x}",
        "sarcastic": lambda x: f"Sure, because that makes total sense... {x}"
    }
    return personas.get(style, personas["default"])(prompt)
