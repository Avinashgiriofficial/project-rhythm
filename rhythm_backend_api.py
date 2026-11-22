import os
import re
import hmac
import json
import time
import base64
import hashlib
import secrets
import sqlite3
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

app = FastAPI(title="Project Rhythm — Adaptive AI Guardian API")

# =========================================================
# STORAGE LAYER: POSTGRESQL (SUPABASE/NEON) WITH SQLITE FALLBACK
# =========================================================

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip().strip('"').strip("'")
USE_POSTGRES = bool(DATABASE_URL and psycopg2 is not None)
SQLITE_PATH = os.environ.get("SQLITE_PATH", "rhythm_pilot.db")


def get_db_conn():
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    identifier TEXT UNIQUE NOT NULL,
                    id_type TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    password_hash TEXT NOT NULL,
                    created_at BIGINT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_states (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    state_json TEXT NOT NULL,
                    updated_at BIGINT NOT NULL
                );
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT UNIQUE NOT NULL,
                    id_type TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_states (
                    user_id INTEGER PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
        conn.commit()
    finally:
        conn.close()


init_db()


# =========================================================
# AUTH HELPERS: PBKDF2 PASSWORD HASHING & HMAC BEARER TOKENS
# =========================================================

def _auth_secret() -> bytes:
    secret = (
        os.environ.get("AUTH_SECRET")
        or os.environ.get("GEMINI_API_KEY")
        or "rhythm-pilot-dev-secret-change-in-render"
    ).strip()
    return secret.encode("utf-8")


def normalize_identifier(raw: str) -> tuple[str, str]:
    val = (raw or "").strip()
    if "@" in val:
        email = val.lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            raise HTTPException(status_code=400, detail="Please enter a valid email address.")
        return email, "email"
    digits = re.sub(r"[\s\-()]+", "", val)
    if re.match(r"^\+?\d{10,15}$", digits):
        return digits, "phone"
    raise HTTPException(status_code=400, detail="Enter a valid email address or 10-digit phone number.")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, expected_hex = stored.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
        return hmac.compare_digest(dk.hex(), expected_hex)
    except Exception:
        return False


def create_token(user_id: int, identifier: str) -> str:
    payload = {
        "uid": user_id,
        "id": identifier,
        "exp": int(time.time()) + 60 * 60 * 24 * 30,  # 30-day session for pilot
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("utf-8")
    sig = hmac.new(_auth_secret(), raw.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8")
    return f"{raw}.{sig_b64}"


def decode_token(auth_header: Optional[str]) -> dict:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = auth_header.split(" ", 1)[1].strip()
    try:
        raw, sig_b64 = token.split(".", 1)
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(_auth_secret(), raw.encode("utf-8"), hashlib.sha256).digest()
        ).decode("utf-8")
        if not hmac.compare_digest(sig_b64, expected_sig):
            raise ValueError("Bad signature")
        payload = json.loads(base64.urlsafe_b64decode(raw.encode("utf-8")).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            raise ValueError("Session expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.")


class AuthSignupRequest(BaseModel):
    identifier: str
    password: str = Field(min_length=6, max_length=128)
    name: str = ""


class AuthLoginRequest(BaseModel):
    identifier: str
    password: str


class UserStatePayload(BaseModel):
    state: dict


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
    diary_sync_enabled: bool = False
    recent_diary_entries: List[dict] = Field(default_factory=list)


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
5. OPTIONAL DIARY CONTEXT: Only when `diary_sync_enabled` is true and `recent_diary_entries` are provided, use those diary reflections gently and respectfully to offer deeper emotional support without sounding intrusive. Never quote private entries verbatim unless helpful.
6. TONE & SAFETY:
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
    api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on Render.")
    return genai.Client(api_key=api_key)


def get_active_models() -> List[str]:
    pinned = os.environ.get("GEMINI_MODEL", "").strip()
    defaults = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
    ]
    return [pinned] + [m for m in defaults if m != pinned] if pinned else defaults


def key_looks_valid(raw_key: str) -> bool:
    return (raw_key.startswith("AIza") or raw_key.startswith("AQ.")) and len(raw_key) >= 24


def call_gemini_with_retry(client: genai.Client, prompt: str, system_instruction: str, schema):
    models_to_try = get_active_models()
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
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                err_str = str(e)
                print(f"[Rhythm AI] {model_name} structured call failed: {err_str}")
                if "404" in err_str or "NOT_FOUND" in err_str:
                    break
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    time.sleep(1.5)
                    break
                try:
                    fallback_prompt = (
                        f"{prompt}\n\nReturn ONLY valid JSON matching this schema:\n"
                        f"{json.dumps(schema.model_json_schema())}"
                    )
                    resp2 = client.models.generate_content(
                        model=model_name,
                        contents=fallback_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.5,
                        ),
                    )
                    return schema.model_validate_json(resp2.text)
                except Exception as e2:
                    last_err = e2
                    print(f"[Rhythm AI] {model_name} JSON fallback failed: {e2}")
                    break
    raise HTTPException(status_code=502, detail=f"AI service error: {str(last_err)}")


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
    raw_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    return {
        "status": "ok",
        "api_key_configured": bool(raw_key),
        "key_format_looks_valid": key_looks_valid(raw_key),
        "models": get_active_models(),
        "storage_backend": "postgresql" if USE_POSTGRES else "sqlite",
        "auth_ready": True,
    }


# =========================================================
# AUTH & CLOUD STATE SYNC ROUTES
# =========================================================

def _ph() -> str:
    return "%s" if USE_POSTGRES else "?"


def _fetch_user_state(cur, user_id: int) -> dict:
    cur.execute(f"SELECT state_json FROM user_states WHERE user_id = {_ph()}", (user_id,))
    row = cur.fetchone()
    if not row:
        return {}
    raw = row["state_json"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]
    try:
        return json.loads(raw)
    except Exception:
        return {}


@app.post("/api/auth/signup")
def auth_signup(req: AuthSignupRequest):
    identifier, id_type = normalize_identifier(req.identifier)
    pw_hash = hash_password(req.password)
    now_ts = int(time.time())
    clean_name = (req.name or "").strip()

    conn = get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else conn.cursor()
        cur.execute(f"SELECT id FROM users WHERE identifier = {_ph()}", (identifier,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="An account with this email or phone already exists. Try signing in.")

        if USE_POSTGRES:
            cur.execute(
                "INSERT INTO users (identifier, id_type, name, password_hash, created_at) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (identifier, id_type, clean_name, pw_hash, now_ts),
            )
            user_id = cur.fetchone()["id"]
            init_state = {"set": {"name": clean_name}}
            cur.execute(
                "INSERT INTO user_states (user_id, state_json, updated_at) VALUES (%s, %s, %s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, json.dumps(init_state), now_ts),
            )
        else:
            cur.execute(
                "INSERT INTO users (identifier, id_type, name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (identifier, id_type, clean_name, pw_hash, now_ts),
            )
            user_id = cur.lastrowid
            init_state = {"set": {"name": clean_name}}
            cur.execute(
                "INSERT OR IGNORE INTO user_states (user_id, state_json, updated_at) VALUES (?, ?, ?)",
                (user_id, json.dumps(init_state), now_ts),
            )

        conn.commit()
        token = create_token(user_id, identifier)
        return {
            "token": token,
            "user": {"id": user_id, "identifier": identifier, "id_type": id_type, "name": clean_name},
            "state": init_state,
        }
    finally:
        conn.close()


@app.post("/api/auth/login")
def auth_login(req: AuthLoginRequest):
    identifier, _ = normalize_identifier(req.identifier)
    conn = get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else conn.cursor()
        cur.execute(
            f"SELECT id, identifier, id_type, name, password_hash FROM users WHERE identifier = {_ph()}",
            (identifier,),
        )
        user = cur.fetchone()
        if not user or not verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect email/phone or password.")

        saved_state = _fetch_user_state(cur, user["id"])
        token = create_token(user["id"], user["identifier"])
        return {
            "token": token,
            "user": {
                "id": user["id"],
                "identifier": user["identifier"],
                "id_type": user["id_type"],
                "name": user["name"] or "",
            },
            "state": saved_state,
        }
    finally:
        conn.close()


@app.get("/api/state")
def get_user_state(authorization: Optional[str] = Header(default=None)):
    payload = decode_token(authorization)
    uid = int(payload["uid"])
    conn = get_db_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else conn.cursor()
        cur.execute(f"SELECT id, identifier, id_type, name FROM users WHERE id = {_ph()}", (uid,))
        user = cur.fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="Account not found.")
        return {
            "user": {
                "id": user["id"],
                "identifier": user["identifier"],
                "id_type": user["id_type"],
                "name": user["name"] or "",
            },
            "state": _fetch_user_state(cur, uid),
        }
    finally:
        conn.close()


@app.put("/api/state")
def save_user_state(body: UserStatePayload, authorization: Optional[str] = Header(default=None)):
    payload = decode_token(authorization)
    uid = int(payload["uid"])
    now_ts = int(time.time())
    state_obj = dict(body.state or {})
    # Respect diary privacy: strip diary pages on the server unless user explicitly opted into AI sync
    diary_sync = ((state_obj.get("set") or {}).get("diarySync")) == "ai_sync"
    if not diary_sync:
        state_obj.pop("diary", None)

    state_str = json.dumps(state_obj)
    student_name = ((state_obj.get("set") or {}).get("name") or "").strip()

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute(
                """
                INSERT INTO user_states (user_id, state_json, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET state_json = EXCLUDED.state_json, updated_at = EXCLUDED.updated_at
                """,
                (uid, state_str, now_ts),
            )
            if student_name:
                cur.execute("UPDATE users SET name = %s WHERE id = %s", (student_name, uid))
        else:
            cur.execute(
                """
                INSERT INTO user_states (user_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE
                SET state_json = excluded.state_json, updated_at = excluded.updated_at
                """,
                (uid, state_str, now_ts),
            )
            if student_name:
                cur.execute("UPDATE users SET name = ? WHERE id = ?", (student_name, uid))
        conn.commit()
        return {"status": "saved", "updated_at": now_ts, "diary_synced": diary_sync}
    finally:
        conn.close()


@app.get("/api/test-ai")
def test_ai():
    """Quick browser check to verify your GEMINI_API_KEY and model connection on Render."""
    raw_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not raw_key:
        return {"status": "error", "detail": "GEMINI_API_KEY is missing in Render Environment Variables."}

    client = genai.Client(api_key=raw_key)
    errors = {}
    for model_name in get_active_models():
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents="Reply with the single word: ONLINE",
            )
            return {
                "status": "ok",
                "model": model_name,
                "reply": resp.text.strip(),
                "key_prefix": raw_key[:6] + "...",
            }
        except Exception as e:
            errors[model_name] = str(e)

    return {
        "status": "error",
        "key_prefix": raw_key[:6] + "...",
        "key_format_looks_valid": key_looks_valid(raw_key),
        "errors": errors,
    }


@app.post("/api/chat", response_model=GuideChatResponse)
def chat_with_guide(req: GuideChatRequest):
    client = get_client()
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