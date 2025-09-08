from google import genai
from google.genai import types

def vsr_text(text, api_key, model):
    
    client = genai.Client(api_key=api_key)

    try:
        generation = client.models.generate_content(
        model=model,
        
        config=types.GenerateContentConfig(
            system_instruction="You are LipReadFixer, a text-restoration editor for noisy, punctuation-free transcripts in English produced by lip-reading VSR models.",
            temperature=0.0),
        
        contents=f"Primary goals (in this order): \
                1. Restore readability: casing, punctuation, sentence/paragraph breaks. \
                2. Correct spelling and grammar. \
                3. Improve coherence and flow while preserving the intended meaning inferred from context. \
                \
                Hard constraints: \
                - Do not invent new facts, names, numbers, dates, or claims not implied by the input or optional context. \
                - Keep the language the same as the input unless explicitly asked to translate.\
                - If something is ambiguous or garbled, pick the most ordinary, context-plausible phrasing. \
                \
                Editing latitude: \
                - You may reorder words, replace terms, merge/split sentences, and add punctuation for clarity. \
                \
                Quality: \
                - Aim for natural, idiomatic phrasing. \
                - Prefer short, clear sentences over long, tangled ones.  \
                \
                INPUT VSR TEXT: {text}"
    )
        
        clean_text = []
        
        if hasattr(generation, "candidates"):
            for cand in generation.candidates:
                if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            clean_text.append(part.text)
                            
        response = "\n".join(clean_text).strip()
                            
        return response
        
    except Exception as e:
        print(f"Error generating content: {e}")
        return None





