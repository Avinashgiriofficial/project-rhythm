import os
import json
import time
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError

app = FastAPI(title="Project Rhythm — Adaptive AI Guardian API")

# =========================================================
# STRUCTURED SCHEMAS FOR SCHEDULE-AWARE CONVERSATIONAL AI
# =========================================================

class ScheduleBlock(BaseModel):
    time: str = Field(description="Formatted time range, e.g. '06:00 AM - 07:00 AM'")
    start: int = Field(description="Start time in minutes from midnight (0-1439)")
    dur: int = Field(description="Duration in minutes, typically 60 or 120")
    activity: str = Field(description="Activity name, e.g. 'Deep Study', 'Breakfast', 'Recovery Break'")
    type: Literal["study", "play", "meal", "break", "flex"] = "flex"
    done: bool = False

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str

class GuideChatRequest(BaseModel):
    student_name: str = ""
    tone: Literal["Gentle", "Balanced", "Direct"] = "Balanced"
    mode: str = "Daily Coach"
    user_message: str
    current_time_str: str = ""
    current_minutes: int = 0
    emotion: str = "Balanced"
    wake: str = "06:00"
    sleep: str = "22:00"
    study_hours_target: int = 5
    play_hours_target: int = 2
    completed_study_hours: float = 0.0
    remaining_study_hours: float = 0.0
    hours_left_in_day: float = 0.0
    backlog_hours: float = 0.0
    current_blocks: List[ScheduleBlock] = Field(default_factory=list)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    learned_memory: List[str] = Field(default_factory=list)
    memory_last_7_days: dict = Field(default_factory=dict)

class GuideChatResponse(BaseModel):
    reply: str = Field(description="Clear, empathetic, specific response addressing the student's exact question, remaining hours, and concrete options.")
    proposed_blocks: Optional[List[ScheduleBlock]] = Field(
        default=None,
        description="If the student asks to rearrange, lighten, shift, or optimize today's schedule, return the complete updated list of blocks for the day (preserving completed blocks and meal anchors). Otherwise null."
    )
    backlog_delta: float = Field(
        default=0.0,
        description="Change to student's study backlog in hours (e.g. +1.0 if 1h of today's study is moved to backlog, -1.0 if 1h of backlog is scheduled into today)."
    )
    action_label: Optional[str] = Field(
        default=None,
        description="Short one-click button label if proposed_blocks or backlog_delta is provided, e.g. 'Apply Rearranged Plan (1h Morning + 1h Evening)'."
    )
    learned_preference: Optional[str] = Field(
        default=None,
        description="If this message reveals a lasting habit or preference (e.g. 'Prefers light 1h morning starts', 'Struggles with afternoon focus'), summarize it in <=10 words to remember for future days. Otherwise null."
    )
    is_crisis: bool = Field(
        default=False,
        description="True ONLY if the user expresses severe distress, hopelessness, or self-harm."
    )

# Backwards-compatible schema for initial daily plan greeting
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

# =========================================================
# SYSTEM INSTRUCTIONS
# =========================================================

CONVERSATIONAL_GUARDIAN_PROMPT = """
You are Rhythm Guide AI, an emotionally intelligent, schedule-aware daily planning companion for students.

YOUR CORE CAPABILITIES:
1. READ THE LIVE SCHEDULE: You are given the student's exact `current_blocks`, `current_time_str`, `completed_study_hours`, `remaining_study_hours`, `hours_left_in_day`, and `backlog_hours`. Always ground your advice in their REAL numbers — never give generic platitudes.
2. REARRANGE SEAMLESSLY: When the student asks to change, lighten, shift, or explore options for their routine (e.g., "change morning 2hr to 1hr", "I missed 2 hours, rearrange my remaining day", "fit my backlog in"):
   - Present 2 clear, practical possibilities in `reply` (e.g., Option A: shift 1h to later today; Option B: keep today lighter and move 1h to Backlog).
   - Provide the BEST recommended rearranged full-day schedule in `proposed_blocks` so they can apply it in ONE click.
   - RULES FOR `proposed_blocks`:
     * Keep all blocks where `done == true` unchanged.
     * Keep all `meal` blocks (Breakfast, Lunch, Dinner) at their exact times.
     * Keep `start` (minutes from midnight), `dur` (minutes, multiples of 60), and `time` ("HH:MM AM - HH:MM PM") mathematically consistent and sorted chronologically within wake..sleep.
     * Set `action_label` to a specific button label like "Apply Plan: 1h Morning + Shift 1h Evening".
3. MANAGE BACKLOG HONESTLY: If the student cannot realistically finish `remaining_study_hours` within `hours_left_in_day`, acknowledge it without guilt, calculate what IS realistically completable today, and set `backlog_delta` (e.g. +1.0) to carry the rest cleanly into their backlog. If they ask to clear backlog today, add a study block and set `backlog_delta` to a negative number (e.g. -1.0).
4. LEARN FROM EVERY INPUT: Look at `learned_memory` to personalize your advice. If the student reveals a new preference or pattern (e.g. "Prefers lighter 1h morning starts"), return a concise `learned_preference` string (max 10 words) so the app remembers it permanently.
5. TONE & SAFETY:
   - Match `tone` (Gentle, Balanced, Direct) and `mode` (Daily Coach, Schedule Reset, Study Strategy, Calm & Focus).
   - Keep `reply` concise, structured with short bullet points when listing options, and warm.
   - If severe distress or self-harm is mentioned, set `is_crisis = true`, `proposed_blocks = null`, and provide calm care with Tele-MANAS (14416) helpline info.
"""

PLAN_GREETING_PROMPT = """
You are Rhythm, an emotionally intelligent daily planning guardian for students.
Reply in 2-3 short, specific sentences: acknowledge their mood/target, ground in their streak and backlog (if any), and give one small next step.
Match the requested tone (Gentle, Balanced, Direct).
If Tired, Stressed, or study_hours >= 7, suggest lightening study load (-1h study, +1 break) with a clear button_label.
"""

def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on Render.")
    return genai.Client(api_key=api_key)

def call_gemini_with_retry(client: genai.Client, prompt: str, system_instruction: str, schema):
    # Use verified production models with automatic fallback
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
    last_err = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.5,
                    ),
                )
                return schema.model_validate_json(response.text)
            except ServerError as e:
                last_err = e
                time.sleep(1.5)
            except Exception as e:
                last_err = e
                break
    raise HTTPException(status_code=502, detail=f"Gemini API error: {str(last_err)}")

# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def serve_frontend():
    for path in ["rhythm_planner.html", "static/rhythm_planner.html", "rhythm.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="rhythm_planner.html not found")

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip())
    }

@app.post("/api/chat", response_model=GuideChatResponse)
def chat_with_guide(req: GuideChatRequest):
    client = get_client()
    # Keep last 8 chat turns for context efficiency
    trimmed_history = [m.model_dump() for m in req.chat_history[-8:]]
    context_payload = req.model_dump()
    context_payload["chat_history"] = trimmed_history

    prompt = (
        f"LIVE STUDENT CONTEXT:\n{json.dumps(context_payload, indent=2)}\n\n"
        f"STUDENT'S LATEST MESSAGE:\n{req.user_message}"
    )
    return call_gemini_with_retry(client, prompt, CONVERSATIONAL_GUARDIAN_PROMPT, GuideChatResponse)

@app.post("/api/guide", response_model=GuideResponse)
def get_guide_response(req: GuideRequest):
    client = get_client()
    prompt = f"Student daily plan context:\n{json.dumps(req.model_dump(), indent=2)}"
    return call_gemini_with_retry(client, prompt, PLAN_GREETING_PROMPT, GuideResponse)
