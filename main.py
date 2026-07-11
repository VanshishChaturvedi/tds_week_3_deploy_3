import os
import json
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DynamicExtractRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(alias="schema") 

@app.post("/dynamic-extract")
def dynamic_extract(payload: DynamicExtractRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Render")
        
    runtime_schema = {
        "type": "OBJECT",
        "properties": {},
        "required": []
    }
    
    for key, field_type in payload.schema_def.items():
        field_type_lower = field_type.lower()
        
        if field_type_lower == "integer":
            target_type = "INTEGER"
        elif field_type_lower == "float":
            target_type = "NUMBER"
        elif field_type_lower == "boolean":
            target_type = "BOOLEAN"
        else:
            target_type = "STRING"
            
        desc = "Must be in ISO format YYYY-MM-DD." if field_type_lower == "date" else ""
        
        runtime_schema["properties"][key] = {
            "type": target_type,
            "description": desc
        }
        runtime_schema["required"].append(key)

    # UPDATED SYSTEM INSTRUCTION: Added Rule #2 specifically for the "Acct 7890" issue
    system_instruction = (
        "You are a strict JSON data extraction API. Your job is to extract exact substrings.\n"
        "CRITICAL RULES:\n"
        "1. EXACT STRING MATCHING: When extracting text, copy the exact characters. Do NOT add missing articles (like 'a', 'an', 'the'). Do NOT fix grammar. Do NOT add periods at the end.\n"
        "2. ENTITY NAMES ONLY: If extracting a name (like a bank, vendor, or store), extract ONLY the core name. Do NOT include account numbers, IDs, or extra context (e.g., if the text says 'HDFC Acct 7890', return exactly 'HDFC').\n"
        "3. EXAMPLE: If extracting an issue and the text says 'laptop arrived damaged', return exactly 'laptop arrived damaged'. NEVER return 'The laptop arrived damaged.'\n"
        "4. Return exactly the keys requested. No extra keys, no missing keys.\n"
        "5. If a field cannot be found, you MUST set its value to null.\n"
        "6. Dates must be formatted as YYYY-MM-DD.\n"
        "7. Integers/floats must be valid JSON numbers, not strings."
    )

    url = "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash:generateContent"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload_data = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "parts": [{"text": payload.text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": runtime_schema
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload_data)
        response.raise_for_status() 
        
        data = response.json()
        
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Clean up Markdown blocks if Gemini included them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
            
        raw_text = raw_text.strip()
        return json.loads(raw_text)
        
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, requests.exceptions.HTTPError):
            error_msg += f" - Response Text: {response.text}"
        print(f"CRITICAL EXTRACTION ERROR: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
