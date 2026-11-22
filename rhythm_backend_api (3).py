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
    time: str = Field(default="", description="Formatted time range, e.g. '06:00 AM - 07:00 AM'")
    start: int = Field(default=360, description="Start time in minutes from midnight (0-1439)")
    dur: int = Field(default=60, description="Duration in minutes, typically 60 or 120")
    activity: str = Field(default="Focus Block", description="Activity name")
    type: Literal["study", "play", "meal", "break", "flex"] = "flex"
    done: bool = False


class ChatMessage(BaseModel):
    role: str = "user"
    text: str = ""


class GuideChatRequest(BaseModel):
    student_name: str = ""
    tone: str = "Balanced"
    mode: str = "Daily Coach"
    user_message: str
    current_time_str: str = ""
    current_minutes: int = 0
    emotion: str = "Balanced"
    wake: str = "06:00"
    sleep: str = "22:00"
    study_hours_target: float = 5.0
    play_hours_target: float = 2.0
    completed_study_hours: float = 0.0
    remaining_study_hours: float = 0.0
    hours_left_in_day: float = 0.0
    backlog_hours: float = 0.0
    current_blocks: List[dict] = Field(default_factory=list)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    learned_memory: List[str] = Field(default_factory=list)
    memory_last_7_days: dict = Field(default_factory=dict)
    diary_sync_enabled: bool = False
    recent_diary_entries: List[dict] = Field(default_factory=list)


class GuideChatResponse(BaseModel):
    reply: str
    proposed_blocks: Optional[List[ScheduleBlock]] = None
    backlog_delta: float = 0.0
    action_label: Optional[str] = None
    learned_preference: Optional[str] = None
    is_crisis: bool = False


class SuggestedAction(BaseModel):
    type: Literal["none", "adjust_load", "add_break", "crisis_support"] = "none"
    study_hours_delta: int = 0
    extra_breaks: int = 0
    button_label: Optional[str] = None


class GuideResponse(BaseModel):
    reply: str
    suggested_action: SuggestedAction = Field(default_factory=SuggestedAction)


class GuideRequest(BaseModel):
    student_name: str = ""
    tone: str = "Balanced"
    today: dict
    memory_last_7_days: dict = Field(default_factory=dict)


# =========================================================
# SYSTEM INSTRUCTIONS & JSON TEMPLATES
# =========================================================

CONVERSATIONAL_GUARDIAN_PROMPT = """
You are Rhythm Guide AI, a fast, warm, schedule-aware daily planning companion for students.

RULES:
1. GROUND IN LIVE DATA: Use the student's real remaining_study_hours, hours_left_in_day, backlog_hours, and current_blocks. Keep replies crisp (2-4 short sentences or 2-3 brief bullets).
2. REARRANGE ONLY WHEN ASKED:
   - For normal chat, status checks, motivation, or study advice, ALWAYS set "proposed_blocks": null.
   - ONLY populate "proposed_blocks" when the student explicitly asks to rearrange, split, shift, or modify their schedule.
   - When returning "proposed_blocks": keep done==true blocks and all "meal" blocks unchanged; keep start (mins from midnight), dur (multiples of 60), and time ("HH:MM AM - HH:MM PM") accurate and sorted; valid types: "study","play","meal","break","flex"; include a short "action_label".
3. BACKLOG: If remaining_study_hours > hours_left_in_day or student asks to move/clear backlog, set numeric "backlog_delta" (+ to add, - to clear).
4. MEMORY & DIARY: Use learned_memory subtly. Only set "learned_preference" (<=10 words) if a brand-new habit is revealed. Use recent_diary_entries gently only when diary_sync_enabled is true.
5. SAFETY: On severe distress/self-harm, set "is_crisis": true, "proposed_blocks": null, and share Tele-MANAS (14416) warmly.

RETURN ONLY ONE COMPACT VALID JSON OBJECT:
{"reply":"string","proposed_blocks":null,"backlog_delta":0.0,"action_label":null,"learned_preference":null,"is_crisis":false}
"""

PLAN_GREETING_PROMPT = """
You are Rhythm, a calm daily planning guardian for students.
Reply in 2 short sentences acknowledging their mood, streak, and backlog (if any) in the requested tone (Gentle, Balanced, Direct).
If Tired, Stressed, Angry, or study_hours >= 7, suggest lightening study load (-1h study, +1 break).

RETURN ONLY ONE COMPACT VALID JSON OBJECT:
{"reply":"string","suggested_action":{"type":"none","study_hours_delta":0,"extra_breaks":0,"button_label":null}}
"""

_CLIENT: Optional[genai.Client] = None
_CLIENT_KEY: str = ""
_PREFERRED_MODEL: Optional[str] = None


def get_client() -> genai.Client:
    global _CLIENT, _CLIENT_KEY
    api_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on Render.")
    # Reuse a single client instance across requests to avoid repeated TLS/connection setup
    if _CLIENT is None or _CLIENT_KEY != api_key:
        _CLIENT = genai.Client(api_key=api_key)
        _CLIENT_KEY = api_key
    return _CLIENT


def get_active_models() -> List[str]:
    pinned = os.environ.get("GEMINI_MODEL", "").strip()
    defaults = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
    ]
    ordered = [pinned] + [m for m in defaults if m != pinned] if pinned else defaults
    if _PREFERRED_MODEL and _PREFERRED_MODEL in ordered:
        ordered = [_PREFERRED_MODEL] + [m for m in ordered if m != _PREFERRED_MODEL]
    return ordered


def key_looks_valid(raw_key: str) -> bool:
    return (raw_key.startswith("AIza") or raw_key.startswith("AQ.")) and len(raw_key) >= 24


def extract_json_object(raw_text: str) -> Optional[dict]:
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _coerce_block(item: dict) -> Optional[ScheduleBlock]:
    if not isinstance(item, dict):
        return None
    btype = str(item.get("type") or "flex").lower().strip()
    if btype not in ("study", "play", "meal", "break", "flex"):
        btype = "flex"
    try:
        start_val = int(round(float(item.get("start", 360))))
    except Exception:
        start_val = 360
    try:
        dur_val = max(30, int(round(float(item.get("dur", 60)))))
    except Exception:
        dur_val = 60
    return ScheduleBlock(
        time=str(item.get("time") or ""),
        start=start_val,
        dur=dur_val,
        activity=str(item.get("activity") or "Focus Block"),
        type=btype,
        done=bool(item.get("done", False)),
    )


def build_safe_response(raw_text: str, schema):
    data = extract_json_object(raw_text)
    if schema is GuideChatResponse:
        if not data:
            clean = (raw_text or "").strip()
            if not clean:
                raise ValueError("Empty model response")
            return GuideChatResponse(reply=clean)

        raw_blocks = data.get("proposed_blocks")
        blocks = None
        if isinstance(raw_blocks, list) and raw_blocks:
            coerced = [b for b in (_coerce_block(x) for x in raw_blocks) if b is not None]
            blocks = coerced or None

        try:
            delta = float(data.get("backlog_delta") or 0.0)
        except Exception:
            delta = 0.0

        reply_txt = str(data.get("reply") or "").strip() or (raw_text or "").strip()
        return GuideChatResponse(
            reply=reply_txt,
            proposed_blocks=blocks,
            backlog_delta=delta,
            action_label=(str(data["action_label"]).strip() if data.get("action_label") else None),
            learned_preference=(str(data["learned_preference"]).strip() if data.get("learned_preference") else None),
            is_crisis=bool(data.get("is_crisis", False)),
        )

    if schema is GuideResponse:
        if not data:
            clean = (raw_text or "").strip()
            if not clean:
                raise ValueError("Empty model response")
            return GuideResponse(reply=clean, suggested_action=SuggestedAction())

        act_raw = data.get("suggested_action") if isinstance(data.get("suggested_action"), dict) else {}
        act_type = str(act_raw.get("type") or "none").strip()
        if act_type not in ("none", "adjust_load", "add_break", "crisis_support"):
            act_type = "none"
        try:
            st_delta = int(round(float(act_raw.get("study_hours_delta") or 0)))
        except Exception:
            st_delta = 0
        try:
            ex_brk = int(round(float(act_raw.get("extra_breaks") or 0)))
        except Exception:
            ex_brk = 0

        reply_txt = str(data.get("reply") or "").strip() or (raw_text or "").strip()
        return GuideResponse(
            reply=reply_txt,
            suggested_action=SuggestedAction(
                type=act_type,
                study_hours_delta=st_delta,
                extra_breaks=ex_brk,
                button_label=(str(act_raw["button_label"]).strip() if act_raw.get("button_label") else None),
            ),
        )

    raise ValueError(f"Unsupported schema: {schema}")


def call_gemini_with_retry(
    client: genai.Client,
    prompt: str,
    system_instruction: str,
    schema,
    max_tokens: int = 900,
):
    global _PREFERRED_MODEL
    models_to_try = get_active_models()
    last_err = None

    for model_name in models_to_try:
        # Step 1: Fast JSON-mode call with output token cap
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.4,
                        max_output_tokens=max_tokens,
                    ),
                )
                result = build_safe_response(resp.text or "", schema)
                _PREFERRED_MODEL = model_name
                return result
            except ServerError as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                err_str = str(e)
                print(f"[Rhythm AI] {model_name} JSON call failed: {err_str}")
                if "404" in err_str or "NOT_FOUND" in err_str:
                    break
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    time.sleep(0.6)
                    break

                # Step 2: Plain call fallback
                try:
                    plain_resp = client.models.generate_content(
                        model=model_name,
                        contents=f"{system_instruction}\n\n{prompt}",
                        config=types.GenerateContentConfig(
                            temperature=0.4,
                            max_output_tokens=max_tokens,
                        ),
                    )
                    result = build_safe_response(plain_resp.text or "", schema)
                    _PREFERRED_MODEL = model_name
                    return result
                except Exception as e2:
                    last_err = e2
                    print(f"[Rhythm AI] {model_name} plain fallback failed: {e2}")
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


@app.get("/api/test-chat")
def test_chat():
    """Quick browser check for the full /api/chat JSON pipeline."""
    client = get_client()
    sample_prompt = (
        "LIVE STUDENT CONTEXT:\n{\"student_name\":\"Test\",\"remaining_study_hours\":2,\"hours_left_in_day\":4}\n\n"
        "STUDENT'S LATEST MESSAGE:\nHi! Say hello in 1 short sentence."
    )
    res = call_gemini_with_retry(client, sample_prompt, CONVERSATIONAL_GUARDIAN_PROMPT, GuideChatResponse)
    return {"status": "ok", "result": res.model_dump()}


@app.post("/api/chat", response_model=GuideChatResponse)
def chat_with_guide(req: GuideChatRequest):
    client = get_client()
    # Keep context compact so Gemini processes input tokens & responds faster
    trimmed_history = [
        {"role": m.role, "text": (m.text or "")[:280]}
        for m in req.chat_history[-6:]
    ]
    compact_ctx = {
        "name": req.student_name,
        "tone": req.tone,
        "mode": req.mode,
        "now": req.current_time_str,
        "now_min": req.current_minutes,
        "mood": req.emotion,
        "wake": req.wake,
        "sleep": req.sleep,
        "study_target": req.study_hours_target,
        "play_target": req.play_hours_target,
        "done_study": req.completed_study_hours,
        "rem_study": req.remaining_study_hours,
        "hrs_left": req.hours_left_in_day,
        "backlog": req.backlog_hours,
        "blocks": req.current_blocks,
        "history": trimmed_history,
        "learned_memory": req.learned_memory[:8],
        "stats_7d": req.memory_last_7_days,
    }
    if req.diary_sync_enabled and req.recent_diary_entries:
        compact_ctx["diary_sync_enabled"] = True
        compact_ctx["recent_diary_entries"] = req.recent_diary_entries[:2]

    # Allocate higher token cap only when student appears to want a schedule edit
    wants_rearrange = bool(
        re.search(r"\b(rearrange|arrange|shift|split|change|move|reschedule|lighten|fit|option|table|schedule)\b", req.user_message, re.I)
    )
    token_budget = 1100 if wants_rearrange else 380

    prompt = (
        f"CONTEXT:{json.dumps(compact_ctx, separators=(',', ':'))}\n"
        f"MESSAGE:{req.user_message}"
    )
    return call_gemini_with_retry(
        client, prompt, CONVERSATIONAL_GUARDIAN_PROMPT, GuideChatResponse, max_tokens=token_budget
    )


@app.post("/api/guide", response_model=GuideResponse)
def get_guide_response(req: GuideRequest):
    client = get_client()
    prompt = f"CONTEXT:{json.dumps(req.model_dump(), separators=(',', ':'))}"
    return call_gemini_with_retry(
        client, prompt, PLAN_GREETING_PROMPT, GuideResponse, max_tokens=220
    )