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
    # Grabs the AI Pipe token from Render Environment Variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in Render")
        
    # 1. Build the schema dictionary
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

    system_instruction = (
        "You are a strict data extraction API. Extract information from the text exactly matching the provided schema.\n"
        "CRITICAL RULES:\n"
        "1. Return exactly the keys requested. No extra keys, no missing keys.\n"
        "2. If a field's value cannot be definitively found in the text, you MUST set its value to null.\n"
        "3. Dates must be formatted as YYYY-MM-DD.\n"
        "4. Integers and floats must be valid JSON numbers, not strings.\n"
        "5. EXACT EXTRACTION: Extract the exact raw phrase from the source text. Do NOT add periods, do NOT alter capitalization, and do NOT add conversational filler like 'The' to make it a complete sentence."
    )

    # 2. EXACT AI Pipe URL and Headers from their documentation
    url = "https://aipipe.org/geminiv1beta/models/gemini-1.5-flash:generateContent"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 3. Construct the exact REST payload for Gemini
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
        # 4. Make the direct request, bypassing SDK routing issues
        response = requests.post(url, headers=headers, json=payload_data)
        response.raise_for_status() 
        
        data = response.json()
        
        # Extract the text from the response structure
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
