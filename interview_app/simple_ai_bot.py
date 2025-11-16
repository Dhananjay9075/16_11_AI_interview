"""
Simple AI Bot - Direct port from app.py for Django
This is a clean implementation that mirrors the Flask app.py exactly
"""
import os
import uuid
import time
from typing import Dict, List
import google.generativeai as genai
from google.cloud import texttospeech
from django.conf import settings

# Configure Gemini - Get API key from Django settings
api_key = getattr(settings, 'GEMINI_API_KEY', '')
if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ WARNING: GEMINI_API_KEY not set. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env file")

# Google Cloud TTS credentials (optional)
credentials_path = os.path.join(settings.BASE_DIR, "ringed-reach-471807-m3-cf0ec93e3257.json")
if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    print("✅ Google Cloud credentials found")
else:
    print("⚠️ Google Cloud credentials not found - TTS will be disabled")

# Upload directory for audio files
UPLOADS_DIR = os.path.join(settings.MEDIA_ROOT, 'ai_uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)


class InterviewSession:
    """Exact copy from app.py"""
    def __init__(self, session_id, candidate_name, jd_text, max_questions=4):
        self.session_id = session_id
        self.candidate_name = candidate_name
        self.conversation_history = []
        self.current_question_number = 0
        self.max_questions = max_questions
        self.is_completed = False
        self.jd_text = jd_text
        self.asked_for_questions = False
        self.completed_at = None
        self.awaiting_answer = False
        self.last_active_question_text = ""
        self.question_asked_at = 0.0
        self.first_voice_at = None
        self.last_transcript_update_at = None
        self.last_transcript_word_count = 0
        self.initial_silence_action_taken = False
    
    def add_interviewer_message(self, text):
        self.conversation_history.append({"role": "interviewer", "text": text})
    
    def add_candidate_message(self, text):
        self.conversation_history.append({"role": "candidate", "text": text})
    
    def get_last_interviewer_question(self):
        for message in reversed(self.conversation_history):
            if message["role"] == "interviewer":
                return message["text"]
        return None
    
    def get_conversation_context(self):
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in self.conversation_history])


# Global sessions storage (just like app.py)
sessions = {}


def gemini_generate(prompt):
    """Use gemini-2.0-flash for interview questions"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text if resp and resp.text else "Could not generate a response."
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"[Gemini error: {str(e)}]"


def text_to_speech(text, filename):
    """Exact copy from app.py - Generate MP3 file with fallback for missing credentials"""
    try:
        print(f"🎤 TTS: Generating audio for text: {text[:100]}...")
        if not filename.lower().endswith('.mp3'):
            filename = f"{os.path.splitext(filename)[0]}.mp3"

        # Check if credentials are available
        if not os.path.exists(credentials_path):
            print("⚠️ TTS: Google Cloud credentials not available - returning empty URL")
            return ""

        print(f"🎤 TTS: Creating Google Cloud TTS client...")
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        # Get male en-IN voice
        try:
            available_voices = client.list_voices(language_code="en-IN").voices
        except Exception:
            available_voices = []

        def score_voice(v):
            name = getattr(v, 'name', '') or ''
            quality = 2 if 'Neural2' in name else (1 if 'Wavenet' in name else 0)
            return (quality, name)

        male_voices = [v for v in available_voices if v.ssml_gender == texttospeech.SsmlVoiceGender.MALE]
        selected_voice_name = None
        if male_voices:
            male_voices.sort(key=score_voice, reverse=True)
            selected_voice_name = male_voices[0].name

        if selected_voice_name:
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                name=selected_voice_name
            )
        else:
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-IN",
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        audio_path = os.path.join(UPLOADS_DIR, filename)
        with open(audio_path, "wb") as out:
            out.write(response.audio_content)
        
        # Return URL
        audio_url = f"{settings.MEDIA_URL}ai_uploads/{filename}"
        print(f"✅ TTS: Generated audio URL: {audio_url}")
        return audio_url
    except Exception as e:
        print(f"❌ Google Cloud TTS error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_question(session, question_type="introduction", last_answer_text=None):
    """Exact copy from app.py"""
    # Simple JD context (no RAG for now)
    jd_context = session.jd_text[:500] if session.jd_text else "Technical Role"
    conversation_context = session.get_conversation_context()
    
    if question_type == "introduction":
        prompt = (
            f"You are a professional interviewer. The candidate's name is {session.candidate_name}.\n\n"
            f"Job Description Context: {jd_context}\n\n"
            "Ask a warm, professional introduction question to start the interview. "
            "Keep it conversational and friendly. Ask only one concise, single-line question."
        )
    elif question_type == "follow_up":
        prompt = (
            "You are a professional technical interviewer conducting a technical interview. Ask only single-line, direct questions.\n\n"
            f"Job Description Context: {jd_context}\n\n"
            "Interview so far:\n"
            f"{conversation_context}\n\n"
            "Based on the candidate's last answer, ask a relevant follow-up question grounded in the JD. "
            "IMPORTANT: You may use a brief introductory phrase like 'That's okay' or 'I understand' ONCE, but DO NOT repeat it. "
            "NEVER say phrases like 'That's okay, that's okay' or 'That's fine, that's fine' - this is repetitive and unprofessional. "
            "If you use an introductory phrase, use it only ONCE, then immediately ask the question. "
            "If their answer was solid, probe for impact, metrics, trade-offs, or examples. "
            "If it was brief, ask for clarification or a concrete example. Keep it professional and JD-aligned."
        )
    else:  # regular question
        prompt = (
            "You are a professional interviewer conducting an interview. Ask only single-line questions.\n\n"
            f"Job Description Context: {jd_context}\n\n"
            "Interview so far:\n"
            f"{conversation_context}\n\n"
            f"Ask the next interview question (question {session.current_question_number + 1} of {session.max_questions}). "
            "Base your question on the job description and the candidate's latest answer. "
            "Keep it precise, professional, and directly tied to the JD."
        )
    
    return gemini_generate(prompt)


def _reset_question_timers(session):
    """Exact copy from app.py"""
    session.question_asked_at = time.time()
    session.first_voice_at = None
    session.last_transcript_update_at = None
    session.last_transcript_word_count = 0
    session.initial_silence_action_taken = False


def start_interview(candidate_name: str, jd_text: str, max_questions: int = 4) -> Dict:
    """Exact copy from app.py /start endpoint"""
    try:
        if not jd_text.strip():
            jd_text = "We are looking for a skilled professional with strong technical abilities."
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = InterviewSession(session_id, candidate_name, jd_text, max_questions=max_questions)
        sessions[session_id] = session
        
        print(f"✅ Created session: {session_id}")
        print(f"   Candidate: {candidate_name}")
        print(f"   JD length: {len(jd_text)}")
        
        # Generate introduction question
        question = generate_question(session, "introduction")
        print(f"✅ Generated question: {question[:100]}...")
        
        session.add_interviewer_message(question)
        session.current_question_number += 1
        session.awaiting_answer = True
        session.last_active_question_text = question
        _reset_question_timers(session)
        
        # Generate audio
        audio_url = text_to_speech(question, f"q{session.current_question_number}.mp3")
        print(f"✅ Generated audio: {audio_url}")

        return {
            "session_id": session_id,
            "question": question,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions
        }
    except Exception as e:
        print(f"❌ Error in start_interview: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def upload_answer(session_id: str, transcript: str) -> Dict:
    """Simplified version of app.py /upload_answer"""
    try:
        if not session_id or session_id not in sessions:
            return {"error": "Invalid session ID"}
        
        session = sessions[session_id]
        
        if session.is_completed:
            return {
                "transcript": transcript,
                "completed": True,
                "message": "Interview completed. Thank you for your time!"
            }
        
        # Add candidate message
        transcript = (transcript or "").strip()
        if not transcript:
            transcript = "[No speech detected]"
        session.add_candidate_message(transcript)
        
        print(f"💾 Saved answer: {transcript[:100]}...")
        
        # Check if interview is completed
        if session.current_question_number >= session.max_questions:
            session.completed_at = time.time()
            session.is_completed = True
            return {
                "transcript": transcript,
                "completed": True,
                "message": "Interview completed. Thank you for your time!"
            }
        
        # Generate next question
        if session.current_question_number == 1:
            next_question = generate_question(session, "follow_up", last_answer_text=transcript)
        else:
            next_question = generate_question(session, "regular", last_answer_text=transcript)
        
        print(f"🤖 Generated next question: {next_question[:100]}...")
        
        session.add_interviewer_message(next_question)
        session.current_question_number += 1
        session.awaiting_answer = True
        session.last_active_question_text = next_question
        
        audio_url = text_to_speech(next_question, f"q{session.current_question_number}.mp3")
        _reset_question_timers(session)

        return {
            "transcript": transcript,
            "completed": False,
            "next_question": next_question,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions,
            "continuous": True
        }
    except Exception as e:
        print(f"❌ Error in upload_answer: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

