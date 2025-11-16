import os
import io
import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from django.conf import settings

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional at runtime
    genai = None

try:
    from google.cloud import texttospeech  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    texttospeech = None

# Optional dependencies for RAG
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

import numpy as np


# Configure Gemini - Get API key from Django settings
from django.conf import settings as django_settings
_GEMINI_KEY = getattr(django_settings, 'GEMINI_API_KEY', '')
if genai:
    try:
        genai.configure(api_key=_GEMINI_KEY)
        print(f"✅ Gemini API configured successfully")
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
        pass


# Upload directory for generated audio files
AI_UPLOADS_SUBDIR = "ai_uploads"
AI_UPLOADS_DIR = os.path.join(getattr(settings, "MEDIA_ROOT", "media"), AI_UPLOADS_SUBDIR)
os.makedirs(AI_UPLOADS_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    base = "".join(c for c in name if c.isalnum() or c in ("-", "_", "."))
    return base or uuid.uuid4().hex


class RAGSystem:
    def __init__(self) -> None:
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.model = None
        self.dimension = 384
        self.index = None
        self.jd_chunks: List[str] = []
        self.is_initialized = False

    def process_jd(self, jd_text: str) -> None:
        sentences = [s.strip() for s in jd_text.split(". ") if s.strip()]
        self.jd_chunks = [f"{s}." if not s.endswith(".") else s for s in sentences]

        if self.model is not None and faiss is not None:
            try:
                embeddings = self.model.encode(self.jd_chunks).astype("float32")
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings)
            except Exception:
                self.index = None
        else:
            # Fallback: random vectors to keep flow working
            self.index = None
        self.is_initialized = True

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        if not self.is_initialized:
            return []
        if self.model is not None and self.index is not None and faiss is not None:
            try:
                q = self.model.encode([query]).astype("float32")
                _, idx = self.index.search(q, top_k)
                return [self.jd_chunks[i] for i in idx[0] if 0 <= i < len(self.jd_chunks)]
            except Exception:
                pass
        # Fallback: return first K chunks
        return self.jd_chunks[:top_k]


def _gemini_generate(prompt: str) -> str:
    if not (genai and _GEMINI_KEY):
        return "Could not generate a response."
    try:
        # Use gemini-2.0-flash for interview questions
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or "Could not generate a response."
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"[Gemini error: {str(e)}]"


def _text_to_speech(text: str, filename: str) -> str:
    safe_name = _sanitize_filename(filename)
    if not safe_name.lower().endswith(".mp3"):
        safe_name = f"{os.path.splitext(safe_name)[0]}.mp3"
    out_path = os.path.join(AI_UPLOADS_DIR, safe_name)

    # Prefer Google Cloud TTS if configured
    if texttospeech is not None:
        try:
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE,
            )
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            with open(out_path, "wb") as f:
                f.write(response.audio_content)
            return f"{getattr(settings, 'MEDIA_URL', '/media/').rstrip('/')}/{AI_UPLOADS_SUBDIR}/{safe_name}"
        except Exception:
            pass

    # Graceful fallback: create empty file to keep flow predictable
    try:
        with open(out_path, "wb") as f:
            f.write(b"")
    except Exception:
        pass
    return f"{getattr(settings, 'MEDIA_URL', '/media/').rstrip('/')}/{AI_UPLOADS_SUBDIR}/{safe_name}"


@dataclass
class ChatSession:
    session_id: str
    candidate_name: str
    jd_text: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_question_number: int = 0
    max_questions: int = 4
    is_completed: bool = False
    asked_for_questions: bool = False
    last_active_question_text: str = ""
    question_asked_at: float = 0.0

    def add_interviewer(self, text: str) -> None:
        self.conversation_history.append({"role": "interviewer", "text": text})

    def add_candidate(self, text: str) -> None:
        self.conversation_history.append({"role": "candidate", "text": text})

    def last_interviewer_question(self) -> str:
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "interviewer":
                return msg.get("text", "")
        return ""

    def context_text(self) -> str:
        return "\n".join(f"{m['role']}: {m['text']}" for m in self.conversation_history)


class ChatBotManager:
    def __init__(self) -> None:
        self.sessions: Dict[str, ChatSession] = {}
        self.rag = RAGSystem()

    def _generate_question(self, session: ChatSession, qtype: str = "regular", last_answer: Optional[str] = None) -> str:
        print(f"🔍 Generating {qtype} question for candidate: {session.candidate_name}")
        
        chunks = []
        if last_answer:
            chunks = self.rag.retrieve_context(last_answer, top_k=5)
        if not chunks:
            chunks = self.rag.retrieve_context("job requirements and responsibilities", top_k=5)
        jd_ctx = " ".join(chunks)
        conv = session.context_text()
        
        print(f"🔍 JD context length: {len(jd_ctx)}, conversation length: {len(conv)}")

        if qtype == "introduction":
            prompt = (
                f"You are a professional interviewer. The candidate is {session.candidate_name}.\n\n"
                f"Job Description Context: {jd_ctx}\n\n"
                "Ask a warm, single-line introduction question to start."
            )
        elif qtype == "follow_up":
            prompt = (
                "You are a professional technical interviewer. Ask a concise, single-line, direct follow-up question based on the answer.\n\n"
                f"Job Description Context: {jd_ctx}\n\nInterview so far:\n{conv}\n"
                "IMPORTANT: You may use a brief introductory phrase like 'That's okay' or 'I understand' ONCE, but DO NOT repeat it. "
                "NEVER say phrases like 'That's okay, that's okay' or 'That's fine, that's fine' - this is repetitive and unprofessional. "
                "If you use an introductory phrase, use it only ONCE, then immediately ask the question."
            )
        else:
            prompt = (
                "You are a professional interviewer. Ask the next single-line question grounded in the JD and prior answer.\n\n"
                f"Job Description Context: {jd_ctx}\n\nInterview so far:\n{conv}\n"
            )

        print(f"🔍 Sending prompt to Gemini (length: {len(prompt)})")
        text = _gemini_generate(prompt).strip()
        print(f"🔍 Gemini response: {text[:100]}...")
        return text or "Please describe a recent project you are proud of."

    def start(self, candidate_name: str, jd_text: str, max_questions: int = 4) -> Dict[str, object]:
        if not jd_text.strip():
            return {"error": "Job description is required"}
        self.rag.process_jd(jd_text)
        sid = uuid.uuid4().hex
        session = ChatSession(session_id=sid, candidate_name=candidate_name or "Candidate", jd_text=jd_text, max_questions=max_questions)
        self.sessions[sid] = session

        q = self._generate_question(session, qtype="introduction")
        session.add_interviewer(q)
        session.current_question_number = 1
        session.last_active_question_text = q
        session.question_asked_at = time.time()
        audio_url = _text_to_speech(q, f"q{session.current_question_number}.mp3")

        return {
            "session_id": sid,
            "question": q,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions,
        }

    def upload_answer(self, session_id: str, transcript: str, silence: bool, had_voice: bool) -> Dict[str, object]:
        if not session_id or session_id not in self.sessions:
            return {"error": "Invalid session ID"}
        session = self.sessions[session_id]

        transcript = (transcript or "").strip()
        if not transcript:
            if silence and not had_voice:
                proceed = "I didn’t catch a response — shall we move to the next question?"
                session.add_interviewer(proceed)
                audio_url = _text_to_speech(proceed, f"proceed_{uuid.uuid4().hex}.mp3")
                session.last_active_question_text = proceed
                session.question_asked_at = time.time()
                return {
                    "transcript": "",
                    "completed": False,
                    "next_question": proceed,
                    "audio_url": audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                }
            return {
                "transcript": "",
                "completed": False,
                "acknowledge": True,
                "message": "I didn't catch that. Please try again.",
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
            }

        session.add_candidate(transcript)

        # End when reaching max_questions; otherwise continue
        if session.current_question_number >= session.max_questions:
            session.is_completed = True
            closing = "Interview completed. Thank you for your time!"
            audio_url = _text_to_speech(closing, f"final_{uuid.uuid4().hex}.mp3")
            return {
                "transcript": transcript,
                "completed": True,
                "message": closing,
                "final_audio_url": audio_url,
            }

        qtype = "follow_up" if session.current_question_number == 1 else "regular"
        print(f"🤖 Generating {qtype} question for session {session_id}, current question: {session.current_question_number}")
        next_q = self._generate_question(session, qtype=qtype, last_answer=transcript)
        print(f"🤖 Generated question: {next_q[:100]}...")
        
        if not next_q or next_q.startswith("[Gemini error"):
            print(f"❌ Question generation failed: {next_q}")
            return {
                "transcript": transcript,
                "completed": False,
                "error": f"Failed to generate next question: {next_q}",
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
            }
        
        session.add_interviewer(next_q)
        session.current_question_number += 1
        session.last_active_question_text = next_q
        session.question_asked_at = time.time()
        audio_url = _text_to_speech(next_q, f"q{session.current_question_number}.mp3")
        print(f"✅ Question {session.current_question_number} generated successfully")

        return {
            "transcript": transcript,
            "completed": False,
            "next_question": next_q,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions,
        }

    def repeat(self, session_id: str) -> Dict[str, object]:
        if not session_id or session_id not in self.sessions:
            return {"error": "Invalid session ID"}
        session = self.sessions[session_id]
        last_q = session.last_interviewer_question()
        if not last_q:
            return {"error": "No previous question found"}
        audio_url = _text_to_speech(last_q, f"repeat_{uuid.uuid4().hex}.mp3")
        session.last_active_question_text = last_q
        session.question_asked_at = time.time()
        return {
            "next_question": last_q,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions,
        }

    def transcript_pdf_bytes(self, session_id: str) -> bytes:
        try:
            from fpdf import FPDF  # type: ignore
        except Exception:
            return b""

        if not session_id or session_id not in self.sessions:
            return b""
        session = self.sessions[session_id]

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Interview Transcript", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Candidate: {session.candidate_name}", ln=True)
        pdf.cell(0, 8, f"Session ID: {session.session_id}", ln=True)
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Conversation:", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=12)
        for msg in session.conversation_history:
            role = "Interviewer" if msg.get("role") == "interviewer" else "Candidate"
            text = msg.get("text", "")
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(0, 7, role + ":", align="L")
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 7, text, align="L")
            pdf.ln(2)
        data = pdf.output(dest='S')
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return str(data).encode('latin1', 'replace')


# Singleton manager used by Django views
chatbot_manager = ChatBotManager()


def ai_start_django(candidate_name: str, jd_text: str, session_key: str = None) -> Dict[str, object]:
    """Start AI interview, optionally getting data from database using session_key."""
    from .models import InterviewSession as DjangoInterviewSession
    
    # Get question count from InterviewSlot.ai_configuration or default to 4
    question_count = 4  # Default
    
    # If session_key is provided, get data from database
    if session_key:
        try:
            django_session = DjangoInterviewSession.objects.get(session_key=session_key)
            candidate_name = django_session.candidate_name
            jd_text = django_session.job_description or ""
            
            # Get question count from InterviewSlot.ai_configuration
            try:
                from interviews.models import Interview
                interview = Interview.objects.filter(session_key=session_key).first()
                
                # If not found via session_key, try via candidate email
                if not interview and django_session.candidate_email:
                    from candidates.models import Candidate
                    try:
                        candidate = Candidate.objects.get(email=django_session.candidate_email)
                        interview = Interview.objects.filter(candidate=candidate).order_by('-created_at').first()
                    except:
                        pass
                
                if interview and interview.slot:
                    slot = interview.slot
                    if slot.ai_configuration and isinstance(slot.ai_configuration, dict):
                        question_count = slot.ai_configuration.get('question_count', 4)
                        try:
                            question_count = int(question_count)
                        except (ValueError, TypeError):
                            question_count = 4
                    elif hasattr(slot, 'question_count') and slot.question_count:
                        question_count = int(slot.question_count)
            except Exception:
                # If there's any error getting question count, use default
                pass
        except DjangoInterviewSession.DoesNotExist:
            return {"error": "Invalid session key"}
    
    return chatbot_manager.start(candidate_name, jd_text, max_questions=question_count)


def ai_upload_answer_django(session_id: str, transcript: str, silence: bool, had_voice: bool) -> Dict[str, object]:
    return chatbot_manager.upload_answer(session_id, transcript, silence, had_voice)


def ai_repeat_django(session_id: str) -> Dict[str, object]:
    return chatbot_manager.repeat(session_id)


def ai_transcript_pdf_django(session_id: str) -> bytes:
    return chatbot_manager.transcript_pdf_bytes(session_id)


def get_audio_file_path(filename: str) -> str:
    return os.path.join(AI_UPLOADS_DIR, filename)


