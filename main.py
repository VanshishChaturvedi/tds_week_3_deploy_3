import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

aipipe_token = os.getenv("GEMINI_API_KEY") 
if not aipipe_token:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

# 2. Force the Google SDK to use the AIpipe proxy and headers
client = genai.Client(
    api_key=aipipe_token, 
    http_options={
        'base_url': 'https://aipipe.org',
        'headers': {
            'Authorization': f'Bearer {eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDMyMTVAZHMuc3R1ZHkuaWl0bS5hYy5pbiIsImlhdCI6MTc4Mzc3NDI2MywiaXNzIjoiaHR0cHM6Ly9haXBpcGUub3JnIiwiYXVkIjoiYWlwaXBlLWFwaSIsImV4cCI6MTc4NDM3OTA2M30.k_tnHM6rVkF9ZB-EryUBVrv26HIGRH-sO8hmMfyPEp0}'
        }
    }
)

class DynamicExtractRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(alias="schema") 

@app.post("/dynamic-extract")
async def dynamic_extract(payload: DynamicExtractRequest):
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

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload.text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=runtime_schema,
                system_instruction=system_instruction,
                temperature=0.0 
            )
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("\n", 1)[0]
            
        raw_text = raw_text.strip()
        return json.loads(raw_text)
        
    except Exception as e:
        print(f"CRITICAL EXTRACTION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
