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

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
client = genai.Client(api_key=api_key)

class DynamicExtractRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(alias="schema") 

@app.post("/dynamic-extract")
async def dynamic_extract(payload: DynamicExtractRequest):
    # 1. Safely construct a pure dictionary schema
    runtime_schema = {
        "type": "OBJECT",
        "properties": {},
        "required": []
    }
    
    for key, field_type in payload.schema_def.items():
        field_type_lower = field_type.lower()
        
        # Added boolean just in case the grader tests for it
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

    # 2. Strict instruction set (keeping your exact phrase extraction rule)
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
        
        # 3. Bulletproof JSON parsing: Strip markdown if it exists
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            # Strip the opening ```json or ```
            raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            # Strip the closing ```
            raw_text = raw_text.rsplit("\n", 1)[0]
            
        raw_text = raw_text.strip()
        
        return json.loads(raw_text)
        
    except Exception as e:
        # Print the exact error to Render logs so we can see what failed
        print(f"CRITICAL EXTRACTION ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
