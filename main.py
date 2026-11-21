import os
import json
import time
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError

app = FastAPI(title="Project Rhythm API")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

class SuggestedAction(BaseModel):
    type: Literal["none", "adjust_load", "add_break", "crisis_support"] = "none"
    study_hours_delta: int = 0
    extra_breaks: int = 0
    button_label: Optional[str] = None

class GuideResponse(BaseModel):
    reply: str
    suggested_action: SuggestedAction

class GuideRequest(BaseModel):
    student_name: str = ""
    tone: Literal["Gentle", "Balanced", "Direct"] = "Balanced"
    today: dict
    memory_last_7_days: dict = Field(default_factory=dict)

GUARDIAN_SYSTEM_PROMPT = """
You are Rhythm, an emotionally intelligent daily planning guardian for students.
You are NOT a generic chatbot and NOT a therapist.
1. Reply in 3 short sentences: validate emotion, ground in their schedule/streak, give one small next step.
2. Match the requested tone (Gentle, Balanced, Direct).
3. If Sad, Tired, or Stressed, suggest lightening study load (-1h study, +1 break).
4. If severe distress or self-harm is mentioned, set type="crisis_support", button_label="Call Tele-MANAS (14416)", and respond with calm care and helpline info instead of study advice.
"""

@app.get("/")
def serve_frontend():
    if os.path.exists("rhythm_planner.html"):
        return FileResponse("rhythm_planner.html")
    if os.path.exists("static/rhythm_planner.html"):
        return FileResponse("static/rhythm_planner.html")
    raise HTTPException(status_code=404, detail="rhythm_planner.html not found")

@app.post("/api/guide", response_model=GuideResponse)
def get_guide_response(req: GuideRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")

    prompt = f"Student context:\n{json.dumps(req.model_dump(), indent=2)}"

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=GUARDIAN_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=GuideResponse,
                    temperature=0.6,
                ),
            )
            return GuideResponse.model_validate_json(response.text)
        except ServerError:
            if attempt < 2:
                time.sleep(2)
            else:
                break
        except Exception:
            break

    return GuideResponse(
        reply="I had a quick hiccup connecting, but your plan is ready. Take it one block at a time and start with the easiest task first.",
        suggested_action=SuggestedAction(type="none")
    )
