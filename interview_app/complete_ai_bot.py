"""
COMPLETE AI Bot - Full port from app.py (1678 lines) for Django
This includes ALL prompts, RAG system, and advanced interview logic
"""
import os
import re
import uuid
import time
from typing import Dict, List
import google.generativeai as genai
from django.conf import settings
import numpy as np

# Configure Gemini - Using 2.5-flash as per your app.py
GEMINI_API_KEY = "AIzaSyBu5M6cEckMIRPdttrBTBcRJmTUi5MkpvE"
genai.configure(api_key=GEMINI_API_KEY)

# Google Cloud TTS setup (exactly like app.py)
# Use absolute path from BASE_DIR
credentials_path = os.path.join(settings.BASE_DIR, "ringed-reach-471807-m3-cf0ec93e3257.json")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    print(f"✅ Google Cloud TTS credentials set to: {credentials_path}")
elif os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
    print(f"✅ Google Cloud TTS credentials set to: {GOOGLE_APPLICATION_CREDENTIALS}")
else:
    print(f"⚠️ Google Cloud TTS credentials not found at: {credentials_path} or {GOOGLE_APPLICATION_CREDENTIALS}")

# Import Google Cloud TTS (exactly like app.py)
try:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
    print("✅ Google Cloud TTS imported successfully")
except ImportError as e:
    texttospeech = None
    TTS_AVAILABLE = False
    print(f"⚠️ Google Cloud TTS import failed: {e}")

# Upload directory for audio files
UPLOADS_DIR = os.path.join(settings.MEDIA_ROOT, 'ai_uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ------------------ RAG System (from app.py lines 111-157) ------------------
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS/SentenceTransformer not available - using fallback")


class RAGSystem:
    """Exact copy from app.py"""
    def __init__(self):
        if FAISS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
        self.dimension = 384
        if FAISS_AVAILABLE:
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None
        self.jd_chunks = []
        self.is_initialized = False
    
    def process_jd(self, jd_text):
        """Process job description and create embeddings"""
        # Split JD into chunks
        sentences = jd_text.split('. ')
        self.jd_chunks = [s.strip() + '.' for s in sentences if s.strip()]
        
        if self.model and FAISS_AVAILABLE:
            # Create embeddings using SentenceTransformer
            embeddings = self.model.encode(self.jd_chunks)
            embeddings = embeddings.astype('float32')
            # Reset and rebuild index
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        
        self.is_initialized = True
        print(f"✅ Processed JD into {len(self.jd_chunks)} chunks")
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant JD context"""
        if not self.is_initialized:
            return []
        
        if self.model and FAISS_AVAILABLE and self.index:
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding.astype('float32')
            distances, indices = self.index.search(query_embedding, top_k)
            return [self.jd_chunks[i] for i in indices[0]]
        else:
            # Fallback: return first few chunks
            return self.jd_chunks[:top_k]


# Initialize RAG system
rag_system = RAGSystem()


# ------------------ Interview State Management (from app.py lines 521-558) ------------------
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
        # Timing and activity tracking
        self.question_asked_at = 0.0
        self.first_voice_at = None
        self.last_transcript_update_at = None
        self.last_transcript_word_count = 0
        self.initial_silence_action_taken = False
        self.asked_pre_closing_question = False  # Track if we've asked the question before closing
        self.waiting_for_candidate_question = False  # Track if we're waiting for candidate's question after they said yes
        
        # Question type tracking for ratio management (30% follow-up, 70% regular)
        self.regular_questions_count = 0  # Count of regular questions asked
        self.follow_up_questions_count = 0  # Count of follow-up questions asked
        
        # Process JD for RAG
        rag_system.process_jd(jd_text)
    
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


# Global sessions storage
sessions = {}


# ------------------ Timing Helpers (from app.py lines 562-573) ------------------
def _reset_question_timers(session: InterviewSession):
    session.question_asked_at = time.time()
    session.first_voice_at = None
    session.last_transcript_update_at = None
    session.last_transcript_word_count = 0
    session.initial_silence_action_taken = False


def _count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


# ------------------ Gemini helper (from app.py lines 394-401) ------------------
def gemini_generate(prompt):
    """Use gemini-2.0-flash for interview questions"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text if resp and resp.text else "Could not generate a response."
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"[Gemini error: {str(e)}]"


# ------------------ Helper functions (from app.py lines 403-786) ------------------
def generate_final_closing(session):
    """Generate a brief, warm LLM-based closing for the interview."""
    conversation_context = session.get_conversation_context()
    prompt = (
        "You are a professional interviewer closing an interview.\n\n"
        f"Interview so far (short context):\n{conversation_context[-3000:]}\n\n"
        "Write a very brief, warm closing message. "
        "IMPORTANT: You MUST include 'Thank you for your time' or a similar thank you phrase. "
        "Keep it professional and friendly. "
        "Example: 'Thank you for your time today. We'll be in touch soon.' "
        "Keep it to one or two short sentences."
    )
    closing_text = gemini_generate(prompt)
    # Fallback if generation fails
    if not closing_text or "[Gemini error" in closing_text:
        return "Thank you for your time today. We'll be in touch soon."
    return closing_text


def generate_elaborated_question(session, candidate_request_text: str) -> str:
    """Generate a clearer, more detailed version of the last interviewer question with only 1-2 lines of extra context."""
    last_q = session.get_last_interviewer_question() or ""
    jd_context = " ".join(rag_system.retrieve_context(last_q or candidate_request_text, top_k=5))
    conversation_context = session.get_conversation_context()
    prompt = (
        "You are a professional interviewer. The candidate asked to elaborate/clarify the previous question.\n\n"
        f"Job Description Context: {jd_context}\n\n"
        "Interview so far:\n"
        f"{conversation_context}\n\n"
        f"Original question: {last_q}\n\n"
        "IMPORTANT: Rewrite the question with ONLY 1-2 lines of additional context or clarification. "
        "Add minimal helpful context or a brief example, but DO NOT reveal the answer or provide too much detail. "
        "Keep it concise - the question should still require the candidate to think and provide their own answer. "
        "Do not add multiple questions. Keep it professional and suitable for a technical interview."
    )
    return gemini_generate(prompt)


def is_low_content_answer(text: str) -> bool:
    """Detect very short answers: 4 words or fewer."""
    if not text:
        return True
    try:
        return _count_words(text) <= 4
    except Exception:
        normalized = (text or "").strip()
        return len([w for w in normalized.split() if w]) <= 4


def is_dont_know_answer(text: str) -> bool:
    """Detect if candidate says they don't know or are unsure."""
    if not text:
        return False
    text_lower = text.strip().lower()
    dont_know_phrases = [
        "don't know", "dont know", "do not know", "don't know", "i don't know",
        "i dont know", "i do not know", "not sure", "unsure", "i'm not sure",
        "im not sure", "no idea", "have no idea", "not familiar", "unfamiliar",
        "don't remember", "dont remember", "can't recall", "cant recall",
        "not aware", "unaware", "haven't heard", "havent heard", "never heard",
        "don't understand", "dont understand", "not clear", "unclear",
        "i'm not familiar", "im not familiar", "not my area", "outside my expertise"
    ]
    return any(phrase in text_lower for phrase in dont_know_phrases)


def is_broad_or_vague_answer(answer: str) -> bool:
    """Check if answer is broad or vague (lacks specific details, examples, or depth)."""
    if not answer or not answer.strip():
        return True
    
    # Check for very short answers (4 words or fewer)
    if is_low_content_answer(answer):
        return True
    
    # Check for generic phrases that indicate vague answers
    vague_phrases = [
        "i think", "probably", "maybe", "kind of", "sort of", "basically",
        "generally", "usually", "typically", "in general", "more or less",
        "i guess", "i suppose", "not really", "not much", "nothing specific",
        "just", "only", "a bit", "a little", "somewhat"
    ]
    answer_lower = answer.lower()
    vague_count = sum(1 for phrase in vague_phrases if phrase in answer_lower)
    
    # If answer has multiple vague phrases, it's likely vague
    if vague_count >= 2:
        return True
    
    # Check word count - very short answers are likely vague
    word_count = _count_words(answer)
    if word_count <= 10:
        return True
    
    return False


def answer_matches_jd_context(answer: str, jd_text: str) -> bool:
    """Check if answer topic matches or relates to Job Description context."""
    if not answer or not jd_text:
        return False
    
    # Extract key terms from JD (simple keyword matching)
    jd_lower = jd_text.lower()
    answer_lower = answer.lower()
    
    # Common technical terms that might appear in both
    technical_terms = [
        "python", "java", "javascript", "react", "django", "flask", "node",
        "database", "sql", "api", "rest", "aws", "cloud", "docker", "kubernetes",
        "algorithm", "data structure", "frontend", "backend", "full stack",
        "machine learning", "ai", "testing", "agile", "scrum", "git", "ci/cd"
    ]
    
    # Check if answer contains any technical terms from JD
    matching_terms = [term for term in technical_terms if term in jd_lower and term in answer_lower]
    if len(matching_terms) >= 1:
        return True
    
    # Use RAG system to check relevance
    try:
        relevant_chunks = rag_system.retrieve_context(answer, top_k=3)
        if relevant_chunks:
            # If RAG finds relevant chunks, answer likely matches JD
            return True
    except Exception as e:
        print(f"⚠️ Error checking JD context match: {e}")
    
    return False


def should_ask_follow_up(session: InterviewSession, answer: str) -> bool:
    """
    Determine if a follow-up question should be asked based on:
    1. Answer quality (broad/vague)
    2. JD context match
    3. Ratio constraint (30% follow-up, 70% regular)
    
    IMPORTANT: Candidate questions are NOT counted in the overall question ratio.
    This function is only called for candidate ANSWERS, not candidate QUESTIONS.
    """
    # Don't ask follow-up for closing questions or when handling candidate questions
    if session.asked_for_questions or session.asked_pre_closing_question or session.waiting_for_candidate_question:
        return False
    
    # Don't ask follow-up if the transcript is actually a candidate question
    if is_candidate_question(answer):
        return False
    
    # Calculate current ratio
    total_questions = session.regular_questions_count + session.follow_up_questions_count
    if total_questions > 0:
        current_follow_up_ratio = session.follow_up_questions_count / total_questions
        # If we've already exceeded 30%, don't ask more follow-ups
        if current_follow_up_ratio >= 0.30:
            print(f"📊 Follow-up ratio already at {current_follow_up_ratio*100:.1f}% (target: 30%). Skipping follow-up.")
            return False
        
        # Check projected ratio if we add this follow-up
        projected_follow_ups = session.follow_up_questions_count + 1
        projected_total = total_questions + 1
        projected_ratio = projected_follow_ups / projected_total
        
        # If adding this follow-up would exceed 30%, don't generate it
        if projected_ratio > 0.30:
            print(f"📊 Projected follow-up ratio would be {projected_ratio*100:.1f}% (current: {current_follow_up_ratio*100:.1f}%, target: 30%). Skipping follow-up.")
            return False
    
    # Check if answer is broad/vague
    if not is_broad_or_vague_answer(answer):
        print(f"📊 Answer is detailed/specific. Skipping follow-up.")
        return False
    
    # Check if answer matches JD context
    if not answer_matches_jd_context(answer, session.jd_text):
        print(f"📊 Answer doesn't match JD context. Skipping follow-up.")
        return False
    
    # Both conditions met: answer is broad/vague AND matches JD context
    print(f"✅ Answer is broad/vague and matches JD context. Will ask follow-up.")
    return True


def assess_answer_relevance_with_llm(question: str, answer: str) -> bool:
    """Use LLM to conservatively judge if the answer is related to the question."""
    if not question or not answer:
        return False
    prompt = (
        "You are the interview taker evaluating a candidate's answer for relevance to your question.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Guidelines: Respond 'unrelated' only if the answer is clearly off-topic or non-responsive."
        " If the answer is brief but plausibly responsive, or asks for clarification/repetition, respond 'related'.\n"
        "Answer with exactly one word: related OR unrelated."
    )
    verdict = (gemini_generate(prompt) or "").strip().lower()
    return "related" in verdict and "unrelated" not in verdict


def generate_clarification_prompt_with_question(session) -> str:
    """Generate a one-line polite clarification that includes the original question."""
    original_q = get_last_strict_question(session) or ""
    prompt = (
        "You are the interview taker conducting a live interview. The candidate's answer seems unrelated or unclear.\n"
        f"Original question: {original_q}\n\n"
        "Write exactly one single-line sentence that politely says you didn't understand the answer and asks them to repeat or clarify, "
        "and include the original question after this exact phrase: 'Here is the question again: '. "
        "Do not mention that you didn't understand the question; refer to their answer."
    )
    line = (gemini_generate(prompt) or "").strip().replace("\n", " ")
    return line


def is_elaboration_request(text: str) -> bool:
    """Detect if candidate is asking to elaborate/clarify the question."""
    if not text:
        return False
    t_lower = (text or "").strip().lower()
    cues = [
        "elaborate", "elaboration", "explain", "more detail", "clarify", "clarification",
        "what do you mean", "could you expand", "expand on", "help me understand",
        "can you elaborate", "please elaborate", "say it in detail"
    ]
    return any(cue in t_lower for cue in cues)


def is_repeat_request(text: str) -> bool:
    if not text:
        return False
    t_lower = (text or "").strip().lower()
    cues = [
        "repeat", "again", "can you repeat", "please repeat", "ask again",
        "repeat the question", "ask that question again", "say it again"
    ]
    return any(cue in t_lower for cue in cues)


def is_proceed_prompt_text(text: str) -> bool:
    if not text:
        return False
    t = (text or "").strip().lower()
    cues = [
        "i didn't catch a response", "i didn't catch a response",
        "shall we move to the next question"
    ]
    return any(c in t for c in cues)


def is_affirmative_response(text: str) -> bool:
    if not text:
        return False
    t = (text or "").strip().lower()
    affirmatives = [
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
        "proceed", "move on", "next", "go ahead"
    ]
    return any(t == a or a in t for a in affirmatives)


def is_negative_response(text: str) -> bool:
    if not text:
        return False
    t = (text or "").strip().lower()
    negatives = ["no", "nope", "nah", "not now", "dont", "don't", "do not"]
    return any(t == n or n in t for n in negatives)


def is_candidate_question(text):
    """Heuristically detect if the candidate's transcript is a question for the interviewer."""
    if not text:
        return False
    t = text.strip().lower()
    if "?" in t:
        return True
    question_starters = [
        "what ", "how ", "why ", "when ", "where ", "who ", "which ",
        "can you", "could you", "would you", "should i", "do you", "are you",
        "is it", "may i", "might we", "could i", "please explain", "i have a question",
        "could you explain", "can i ask"
    ]
    return any(t.startswith(qs) or qs in t for qs in question_starters)


def generate_candidate_answer(session, candidate_question_text):
    """Answer the candidate's question about the interview, role, or company using JD context and interview history."""
    jd_context = " ".join(rag_system.retrieve_context(candidate_question_text, top_k=5))
    if not jd_context:
        jd_context = " ".join(rag_system.retrieve_context("job description", top_k=5))
    conversation_context = session.get_conversation_context()

    prompt = (
        "You are the interviewer. The candidate asked a question during the interview about the role, company, or interview process.\n\n"
        f"Job Description Context: {jd_context}\n\n"
        "Interview so far:\n"
        f"{conversation_context}\n\n"
        f"Candidate's question: {candidate_question_text}\n\n"
        "Answer the candidate's question briefly, accurately, and professionally based on the job description and interview context. "
        "If the question is about the role, team, or company, provide a helpful answer. "
        "If the question cannot be answered right now or is not related to the interview, "
        "say: 'Thanks for asking — we'll get back to you via email.' "
        "Keep your answer concise (2-3 sentences maximum). "
        "Do not ask a new interview question in this answer. "
        "IMPORTANT: Do NOT say 'go ahead with your answer' or 'please go ahead' - the candidate has already answered the previous question."
    )
    return gemini_generate(prompt)


def says_no_more_questions(text: str) -> bool:
    """Heuristically detect if candidate indicates they have no more questions."""
    if not text:
        return False
    t = (text or "").strip().lower()
    negatives = [
        "no", "nope", "nah", "that's all", "thats all", "that's it", "thats it",
        "all good", "im good", "i'm good", "i am good", "nothing else", "no further",
        "no more", "no more questions", "no questions", "no question", "none",
        "no thanks", "no thank you", "i'm fine", "im fine", "i am fine"
    ]
    if any(neg in t for neg in negatives):
        return True
    import re as _re
    if _re.search(r"(^|\b)no(\b|$)", t):
        return True
    return False


def looks_like_closing(text: str) -> bool:
    """Detect if text already reads like a final closing message."""
    if not text:
        return False
    tl = text.lower()
    cues = [
        "thank you for your time", "we will be in touch", "we'll be in touch",
        "have a good day", "thanks for your time", "next steps", "we appreciate your time"
    ]
    return any(c in tl for c in cues)


def generate_proceed_prompt(session):
    """Ask politely if the candidate wants to move to the next question."""
    jd_context = " ".join(rag_system.retrieve_context("proceed to next question", top_k=3))
    conversation_context = session.get_conversation_context()
    prompt = (
        "You are a polite interviewer. It seems there was no audible response.\n\n"
        f"Job Description Context: {jd_context}\n\n"
        "Interview so far:\n"
        f"{conversation_context}\n\n"
        "Ask in one concise line: 'I didn't catch a response — shall we move to the next question?'"
        " Avoid extra text."
    )
    return gemini_generate(prompt)


def get_last_strict_question(session: InterviewSession) -> str:
    """Return the most recent interviewer message that looks like a real question."""
    skip_phrases = [
        "thanks for asking", "please go ahead with your answer", "we'll follow up via email",
        "we'll follow up via email", "let's move on", "sure — please go ahead", "sure - please go ahead",
        "i didn't catch a response", "i didn't catch a response", "shall we move to the next question",
        "please try again", "we appreciate your question", "we appreciate your question",
        "before we wrap up"
    ]
    for msg in reversed(session.conversation_history):
        if msg.get("role") != "interviewer":
            continue
        text = (msg.get("text") or "").strip().lower()
        if any(p in text for p in skip_phrases):
            continue
        if msg.get("text", "").strip().endswith("?"):
            return msg["text"]
    return session.get_last_interviewer_question() or ""


# ------------------ TTS (from app.py lines 575-639) ------------------
def text_to_speech(text, filename):
    """Generate MP3 file for given text using Google Cloud Text-to-Speech (from app.py lines 575-639)"""
    try:
        # Ensure filename has .mp3 extension
        if not filename.lower().endswith('.mp3'):
            filename = f"{os.path.splitext(filename)[0]}.mp3"

        if not TTS_AVAILABLE or not texttospeech:
            print("⚠️ TTS library not available - skipping audio")
            return ""

        # Ensure Google Cloud credentials are set
        credentials_path = os.path.join(settings.BASE_DIR, "ringed-reach-471807-m3-cf0ec93e3257.json")
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            print(f"✅ Google Cloud credentials set: {credentials_path}")
        else:
            print(f"❌ Google Cloud credentials not found: {credentials_path}")
            # Still try to proceed in case credentials are set via environment variable

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        # Query available en-IN voices and select a male voice, preferring Neural2 then Wavenet
        try:
            available_voices = client.list_voices(language_code="en-IN").voices
        except Exception as list_err:
            print(f"⚠️ Could not list voices for en-IN: {list_err}")
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
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
        else:
            # Fallback: request male gender without specifying a name
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
        
        # Verify file was created
        if not os.path.exists(audio_path):
            print(f"❌ TTS audio file not created at: {audio_path}")
            return ""
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            print(f"❌ TTS audio file is empty (0 bytes): {audio_path}")
            return ""
        
        # Return URL for Django media serving
        audio_url = f"{settings.MEDIA_URL}ai_uploads/{filename}"
        print(f"✅ TTS audio generated successfully: {audio_url} (size: {file_size} bytes)")
        return audio_url
    except Exception as e:
        print(f"❌ Google Cloud TTS error: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail silently - return empty string so frontend can handle gracefully
        print(f"⚠️ Returning empty audio_url - frontend will handle text-only mode")
        return ""


# ------------------ Question Generation (from app.py lines 641-684) ------------------
def generate_question(session, question_type="introduction", last_answer_text=None):
    """Generate the next interview question using JD + latest transcript context."""
    # Build JD context prioritized by the last answer if available
    jd_context_chunks = []
    if last_answer_text and last_answer_text.strip():
        jd_context_chunks = rag_system.retrieve_context(last_answer_text, top_k=5)
    if not jd_context_chunks:
        jd_context_chunks = rag_system.retrieve_context("job requirements and responsibilities", top_k=5)
    jd_context = " ".join(jd_context_chunks)

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
            "NEVER say 'go ahead with your answer' or 'please go ahead' - the candidate has already answered, so just ask the next question directly. "
            "If you use an introductory phrase, use it only ONCE, then immediately ask the question. "
            "If their answer was solid, probe for impact, metrics, trade-offs, or examples. "
            "If it was brief, ask for clarification or a concrete example. Keep it professional and JD-aligned."
        )
    
    elif question_type == "closing":
        prompt = (
            "You are a professional interviewer wrapping up a technical interview. Ask only single-line questions.\n\n"
            f"Job Description Context: {jd_context}\n\n"
            "Interview so far:\n"
            f"{conversation_context}\n\n"
            "Generate a professional closing question to ask the candidate if they have any questions. "
            "This should be a natural transition to end the interview. "
            "Examples: 'Before we wrap up, do you have any questions for us?' or 'Is there anything you'd like to ask about the role or company?' "
            "Keep it warm, professional, and concise. Ask only one single-line question."
        )
    
    else:  # regular question
        prompt = (
            "You are a professional interviewer conducting an interview. Ask only single-line questions.\n\n"
            f"Job Description Context: {jd_context}\n\n"
            "Interview so far:\n"
            f"{conversation_context}\n\n"
            f"Ask the next interview question (question {session.current_question_number + 1} of {session.max_questions}). "
            "Base your question on the job description and the candidate's latest answer. "
            "IMPORTANT: The candidate has already answered the previous question, so just ask the next question directly. "
            "NEVER say 'go ahead with your answer', 'please go ahead', or similar phrases - they have already answered. "
            "Keep it precise, professional, and directly tied to the JD."
        )
    
    return gemini_generate(prompt)


# ------------------ Routes (from app.py lines 793-830) ------------------
def start_interview(candidate_name: str, jd_text: str, max_questions: int = 4) -> Dict:
    """Exact copy from app.py /start endpoint"""
    try:
        print(f"\n{'='*60}")
        print(f"🤖 START_INTERVIEW CALLED IN complete_ai_bot.py")
        print(f"{'='*60}")
        print(f"   Candidate Name: {candidate_name}")
        print(f"   JD Text Length: {len(jd_text)}")
        print(f"   JD Text Preview: {jd_text[:200]}...")
        print(f"   JD Full Text: {jd_text}")
        print(f"   Max Questions: {max_questions}")
        print(f"{'='*60}\n")
        
        if not jd_text.strip():
            print("❌ ERROR: Job description is empty!")
            return {"error": "Job description is required"}
        
        # Process JD with RAG system
        print(f"📚 Processing JD with RAG system...")
        rag_system.process_jd(jd_text)
        print(f"✅ JD processed with RAG system")
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = InterviewSession(session_id, candidate_name, jd_text, max_questions=max_questions)
        sessions[session_id] = session
        print(f"✅ Session created with ID: {session_id}")
        
        # Generate introduction question
        question = generate_question(session, "introduction")
        session.add_interviewer_message(question)
        session.current_question_number = 1  # Start at 1 (introduction is question 1)
        session.awaiting_answer = True
        session.last_active_question_text = question
        # Introduction is counted as a regular question for ratio tracking
        session.regular_questions_count = 1
        _reset_question_timers(session)
        
        # Generate audio
        audio_url = text_to_speech(question, f"q{session.current_question_number}.mp3")
        
        print(f"🎯 Starting interview with max_questions={session.max_questions}, current_question_number={session.current_question_number}")

        return {
            "session_id": session_id,
            "question": question,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions
        }
    except Exception as e:
        print(f"❌ Error in /start: {e}")
        return {"error": str(e)}


# ------------------ Upload Answer (from app.py lines 832-1259) - COMPLETE VERSION ------------------
def upload_answer(session_id: str, transcript: str, silence_flag: bool = False, had_voice_flag: bool = False) -> Dict:
    """Complete version from app.py /upload_answer endpoint with ALL logic"""
    try:
        if not session_id or session_id not in sessions:
            return {"error": "Invalid session ID"}
        
        session = sessions[session_id]
        
        # If interview already completed, don't accept further answers
        if session.is_completed:
            return {
                "transcript": "[Session already completed]",
                "completed": True,
                "message": "Interview completed. Thank you for your time!"
            }
        
        # Prefer transcript sent from UI
        transcript = (transcript or "").strip()
        print(f"🗒️ /upload_answer: session={session_id}, transcript_len={len(transcript)}")

        if not transcript:
            transcript = "[No speech detected]"
        session.add_candidate_message(transcript)
        print(f"💾 Saved candidate message: '{(transcript[:120] + ('...' if len(transcript) > 120 else ''))}'")
        
        # If we received content, we are no longer awaiting an answer
        if transcript.strip() and not transcript.startswith("["):
            session.awaiting_answer = False
        
        print(f"📶 Client flags: silence={silence_flag}, had_voice={had_voice_flag}")

        # ------------------ Activity tracking & timing-based flow ------------------
        now_ts = time.time()
        current_word_count = _count_words(transcript)
        if current_word_count > session.last_transcript_word_count:
            session.last_transcript_update_at = now_ts
            session.last_transcript_word_count = current_word_count
            if session.first_voice_at is None and current_word_count > 0:
                session.first_voice_at = now_ts

        # If no transcript yet and 4s passed since question asked, auto-advance
        if session.awaiting_answer and not session.initial_silence_action_taken and session.first_voice_at is None:
            if session.question_asked_at and (now_ts - session.question_asked_at) >= 4.0:
                session.initial_silence_action_taken = True
                # For silence, default to regular question (no answer to evaluate for follow-up)
                next_question = generate_question(session, "regular", last_answer_text=transcript)
                session.regular_questions_count += 1
                session.add_interviewer_message(next_question)
                session.current_question_number += 1
                session.awaiting_answer = True
                session.last_active_question_text = next_question
                audio_url = text_to_speech(next_question, f"q{session.current_question_number}.mp3")
                _reset_question_timers(session)
                return {
                    "transcript": transcript or "[No speech detected]",
                    "completed": False,
                    "next_question": next_question,
                    "audio_url": audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }
        
        # If in closing Q&A phase
        if session.asked_for_questions and not session.is_completed:
            transcript_lower = transcript.lower()
            no_resp = (not transcript.strip()) or transcript.startswith("[No speech") or (silence_flag and not had_voice_flag)
            
            # Check if candidate says no or doesn't have questions
            if no_resp or says_no_more_questions(transcript):
                final_text = generate_final_closing(session)
                final_audio = text_to_speech(final_text, f"final_{uuid.uuid4().hex}.mp3")
                session.add_interviewer_message(final_text)
                session.completed_at = time.time()
                session.is_completed = True
                session.waiting_for_candidate_question = False
                return {
                    "transcript": transcript,
                    "completed": True,
                    "final_message": final_text,
                    "final_audio_url": final_audio
                }

            # If we're waiting for the candidate's question (after they said yes)
            if session.waiting_for_candidate_question:
                # Candidate provided their question - answer it
                # IMPORTANT: Candidate questions are NOT counted in the overall question count
                answer_text = generate_candidate_answer(session, transcript)
                session.add_interviewer_message(answer_text)
                session.waiting_for_candidate_question = False
                
                # After answering, ask if they have more questions
                # NOTE: We do NOT increment question_number or question counters for candidate Q&A
                prompt_again = "Do you have any other questions for us?"
                combined = f"{answer_text} {prompt_again}".strip()
                session.add_interviewer_message(combined)
                question_audio_url = text_to_speech(combined, f"closing_{uuid.uuid4().hex}.mp3")
                _reset_question_timers(session)
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": combined,
                    "audio_url": question_audio_url,
                    "question_number": session.current_question_number,  # Keep same question number - not a new interview question
                    "max_questions": session.max_questions,
                    "continuous": True
                }

            # Check if candidate explicitly says NO (only end interview if they say no)
            says_no = says_no_more_questions(transcript)
            
            if says_no:
                # Candidate explicitly said no - end the interview
                final_text = generate_final_closing(session)
                final_audio = text_to_speech(final_text, f"final_{uuid.uuid4().hex}.mp3")
                session.add_interviewer_message(final_text)
                session.completed_at = time.time()
                session.is_completed = True
                session.waiting_for_candidate_question = False
                return {
                    "transcript": transcript,
                    "completed": True,
                    "final_message": final_text,
                    "final_audio_url": final_audio
                }
            
            # Check if candidate says yes or asks a question directly
            says_yes = any(phrase in transcript_lower for phrase in [
                "yes", "yeah", "yep", "sure", "i do", "i have", "i'd like", "i would like"
            ])
            
            # Check if it looks like a direct question (contains question words or ends with ?)
            looks_like_question = any(word in transcript_lower for word in [
                "what", "when", "where", "who", "why", "how", "which", "can", "could", "would", "should"
            ]) or "?" in transcript
            
            if says_yes or looks_like_question:
                if says_yes and not looks_like_question:
                    # Candidate said yes but didn't ask a question yet - ask them what their question is
                    ask_question = "What is your question?"
                    session.add_interviewer_message(ask_question)
                    session.waiting_for_candidate_question = True
                    question_audio_url = text_to_speech(ask_question, f"ask_question_{uuid.uuid4().hex}.mp3")
                    _reset_question_timers(session)
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": ask_question,
                        "audio_url": question_audio_url,
                        "question_number": session.current_question_number,
                        "max_questions": session.max_questions,
                        "continuous": True
                    }
                else:
                    # Candidate asked a question directly - answer it
                    # IMPORTANT: Candidate questions are NOT counted in the overall question count
                    answer_text = generate_candidate_answer(session, transcript)
                    session.add_interviewer_message(answer_text)
                    
                    # After answering, ask if they have more questions
                    # NOTE: We do NOT increment question_number or question counters for candidate Q&A
                    prompt_again = "Do you have any other questions for us?"
                    combined = f"{answer_text} {prompt_again}".strip()
                    session.add_interviewer_message(combined)
                    question_audio_url = text_to_speech(combined, f"closing_{uuid.uuid4().hex}.mp3")
                    _reset_question_timers(session)
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": combined,
                        "audio_url": question_audio_url,
                        "question_number": session.current_question_number,  # Keep same question number - not a new interview question
                        "max_questions": session.max_questions,
                        "continuous": True
                    }
            else:
                # Candidate gave an answer that's not clearly "yes", "no", or a question
                # This might be a response like "thank you", "that's all", or other ambiguous responses
                # Ask them again if they have any questions to clarify
                prompt_again = "Do you have any other questions for us?"
                session.add_interviewer_message(prompt_again)
                question_audio_url = text_to_speech(prompt_again, f"closing_clarify_{uuid.uuid4().hex}.mp3")
                _reset_question_timers(session)
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": prompt_again,
                    "audio_url": question_audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }
        
        # Handle voice commands
        transcript_lower = transcript.lower()
        
        # Proceed-confirmation flow
        last_interviewer = session.get_last_interviewer_question() or ""
        if is_proceed_prompt_text(last_interviewer):
            if is_affirmative_response(transcript):
                # Use ratio logic to decide follow-up vs regular
                should_follow_up = should_ask_follow_up(session, transcript)
                if should_follow_up:
                    next_q_text = generate_question(session, "follow_up", last_answer_text=transcript)
                    session.follow_up_questions_count += 1
                else:
                    next_q_text = generate_question(session, "regular", last_answer_text=transcript)
                    session.regular_questions_count += 1
                session.add_interviewer_message(next_q_text)
                session.current_question_number += 1
                session.awaiting_answer = True
                session.last_active_question_text = next_q_text
                audio_url = text_to_speech(next_q_text, f"q{session.current_question_number}.mp3")
                _reset_question_timers(session)
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": next_q_text,
                    "audio_url": audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }
            elif is_negative_response(transcript):
                last_q = get_last_strict_question(session)
                if last_q:
                    session.add_interviewer_message(last_q)
                    session.awaiting_answer = True
                    session.last_active_question_text = last_q
                    question_audio_url = text_to_speech(last_q, f"q{session.current_question_number}_repeat.mp3")
                    _reset_question_timers(session)
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": last_q,
                        "audio_url": question_audio_url,
                        "question_number": session.current_question_number,
                        "max_questions": session.max_questions,
                        "continuous": True
                    }

        # Repeat command
        if any(word in transcript_lower for word in ["repeat", "again", "can you repeat", "please repeat"]):
            last_q = get_last_strict_question(session)
            if last_q:
                audio_url = text_to_speech(last_q, f"repeat_{uuid.uuid4().hex}.mp3")
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": last_q,
                    "audio_url": audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }

        # Skip command
        if any(phrase in transcript_lower for phrase in ["don't know", "do not know", "skip", "next question", "move on"]):
            # For skip commands, default to regular question (no meaningful answer to evaluate)
            next_q_text = generate_question(session, "regular", last_answer_text=transcript)
            session.regular_questions_count += 1
            # Use the question directly without any reassuring phrases - be professional like a real technical interviewer
            combined_text = next_q_text.strip()
            session.add_interviewer_message(combined_text)
            session.current_question_number += 1
            session.awaiting_answer = True
            session.last_active_question_text = next_q_text
            audio_url = text_to_speech(combined_text, f"q{session.current_question_number}.mp3")
            _reset_question_timers(session)
            return {
                "transcript": transcript,
                "completed": False,
                "next_question": combined_text,
                "audio_url": audio_url,
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
                "continuous": True
            }
        
        # Check if transcript is just noise or empty
        if not transcript.strip() or transcript.startswith("[Speech detected") or transcript.startswith("[No speech detected"):
            if silence_flag and not had_voice_flag:
                proceed_text = generate_proceed_prompt(session)
                audio_url = text_to_speech(proceed_text, f"proceed_{uuid.uuid4().hex}.mp3")
                session.add_interviewer_message(proceed_text)
                _reset_question_timers(session)
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": proceed_text,
                    "audio_url": audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }
            return {
                "transcript": transcript,
                "completed": False,
                "acknowledge": True,
                "message": "I didn't catch that. Please try again.",
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
                "continuous": True
            }

        # Handle unrelated answers
        try:
            last_q_text = get_last_strict_question(session)
        except Exception:
            last_q_text = ""

        # DISABLED: Too aggressive rephrasing - only ask for repeat on explicit commands
        ask_for_repeat = False
        # if is_elaboration_request(transcript) or is_repeat_request(transcript):
        #     ask_for_repeat = False
        # elif last_q_text:
        #     try:
        #         is_related = assess_answer_relevance_with_llm(last_q_text, transcript)
        #         ask_for_repeat = not is_related
        #     except Exception:
        #         ask_for_repeat = False

        if ask_for_repeat:
            clarify_line = generate_clarification_prompt_with_question(session)
            session.add_interviewer_message(clarify_line)
            session.awaiting_answer = True
            session.last_active_question_text = clarify_line
            clarify_audio = text_to_speech(clarify_line, f"clarify_{uuid.uuid4().hex}.mp3")
            _reset_question_timers(session)
            return {
                "transcript": transcript,
                "completed": False,
                "next_question": clarify_line,
                "audio_url": clarify_audio,
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
                "continuous": True
            }
        
        # If the candidate asked a question
        if is_candidate_question(transcript):
            elaboration_cues = [
                "elaborate", "elaboration", "explain", "more detail", "clarify", "clarification",
                "what do you mean", "could you expand", "expand on", "help me understand"
            ]
            repeat_cues = [
                "ask again", "ask that previous", "previous question", "repeat the question",
                "ask that question again", "can you ask that question", "can you ask again"
            ]
            t_lower = transcript.lower()
            wants_elaboration = any(cue in t_lower for cue in elaboration_cues)
            wants_repeat = any(cue in t_lower for cue in repeat_cues)

            if wants_repeat:
                last_q = get_last_strict_question(session)
                if last_q:
                    session.add_interviewer_message(last_q)
                    session.awaiting_answer = True
                    session.last_active_question_text = last_q
                    question_audio_url = text_to_speech(last_q, f"q{session.current_question_number}_repeat.mp3")
                    _reset_question_timers(session)
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": last_q,
                        "audio_url": question_audio_url,
                        "question_number": session.current_question_number,
                        "max_questions": session.max_questions,
                        "continuous": True
                    }
            if wants_elaboration:
                elaborated = generate_elaborated_question(session, transcript)
                session.add_interviewer_message(elaborated)
                session.awaiting_answer = True
                session.last_active_question_text = elaborated
                elaborated_audio_url = text_to_speech(elaborated, f"q{session.current_question_number}_elab.mp3")
                _reset_question_timers(session)
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": elaborated,
                    "audio_url": elaborated_audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }

            # Answer the candidate question - respond to their question about the interview/role
            # IMPORTANT: Candidate questions are NOT counted in the overall question count
            answer_text = generate_candidate_answer(session, transcript)
            session.add_interviewer_message(answer_text)

            # After answering candidate's question, continue with the interview
            # Ask them to proceed with answering the current question (if there is one) or move to next
            # NOTE: We do NOT increment question_number or question counters for candidate Q&A
            if session.last_active_question_text and not session.awaiting_answer:
                # There's a pending question - ask them to answer it
                prompt_text = f"Thanks for asking. {session.last_active_question_text}"
            else:
                # No pending question - continue with interview
                prompt_text = "Thanks for asking. Let's continue with the interview."
            
            session.add_interviewer_message(prompt_text)
            session.awaiting_answer = True
            session.last_active_question_text = prompt_text
            
            # Return both answer and prompt
            return {
                "transcript": transcript,
                "completed": False,
                "interviewer_answer": answer_text,
                "answer_audio_url": text_to_speech(answer_text, f"ans_{uuid.uuid4().hex}.mp3"),
                "next_question": prompt_text,
                "audio_url": text_to_speech(prompt_text, f"prompt_{uuid.uuid4().hex}.mp3"),
                "question_number": session.current_question_number,
                "max_questions": session.max_questions,
                "continuous": True
            }

        # Check if interview is completed (regular flow)
        # max_questions represents the TOTAL number of questions including pre-closing and closing
        # So if max_questions is 4, we ask: 2 main + 1 pre-closing + 1 closing = 4 total
        # The introduction question is question 1, so we need to account for that
        # Calculate how many main questions we should ask AFTER introduction (max_questions - 3: intro + pre-closing + closing)
        # Example: if max_questions=4, we want: intro(1) + 1 main(2) + pre-closing(3) + closing(4) = 4 total
        main_questions_after_intro = max(0, session.max_questions - 3)  # Questions after intro, before pre-closing
        
        # Check if we've asked enough main questions (intro is 1, so we check if current > 1 + main_questions_after_intro)
        main_questions_limit = 1 + main_questions_after_intro  # Total main questions including intro
        
        print(f"📊 Question count check: current={session.current_question_number}, max={session.max_questions}, main_limit={main_questions_limit}, main_after_intro={main_questions_after_intro}")
        
        if session.current_question_number >= main_questions_limit:
            # First, ask one more question before the closing question (pre-closing question)
            if not session.asked_pre_closing_question:
                # Check if candidate gave a "don't know" answer - if so, skip pre-closing and go directly to closing
                if is_dont_know_answer(transcript):
                    print(f"⚠️ Candidate gave 'don't know' answer at pre-closing stage, skipping pre-closing question and going directly to closing")
                    session.asked_pre_closing_question = True  # Mark as asked so we skip it
                    # Go directly to closing question
                    closing = generate_question(session, "closing", last_answer_text=transcript)
                    session.add_interviewer_message(closing)
                    session.current_question_number += 1
                    session.awaiting_answer = True
                    session.last_active_question_text = closing
                    session.asked_for_questions = True
                    _reset_question_timers(session)
                    closing_audio = text_to_speech(closing, f"closing_{uuid.uuid4().hex}.mp3")
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": closing,
                        "audio_url": closing_audio,
                        "question_number": session.current_question_number,
                        "max_questions": session.max_questions,
                        "continuous": True
                    }
                
                # Generate one more regular question before closing (pre-closing question)
                next_q_text = generate_question(session, "regular", last_answer_text=transcript)
                session.add_interviewer_message(next_q_text)
                session.current_question_number += 1
                session.regular_questions_count += 1  # Pre-closing is a regular question
                session.awaiting_answer = True
                session.last_active_question_text = next_q_text
                session.asked_pre_closing_question = True
                _reset_question_timers(session)
                question_audio_url = text_to_speech(next_q_text, f"q{session.current_question_number}.mp3")
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": next_q_text,
                    "audio_url": question_audio_url,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,  # Total questions including pre-closing and closing
                    "continuous": True
                }
            # After pre-closing question is answered, ask the closing question (generated dynamically)
            elif not session.asked_for_questions:
                # Check if candidate gave a "don't know" answer to pre-closing question
                # If so, go directly to closing without asking another question
                if is_dont_know_answer(transcript):
                    print(f"⚠️ Candidate gave 'don't know' answer to pre-closing question, going directly to closing")
                    # Go directly to closing question
                    closing = generate_question(session, "closing", last_answer_text=transcript)
                    session.add_interviewer_message(closing)
                    session.current_question_number += 1
                    session.awaiting_answer = True
                    session.last_active_question_text = closing
                    session.asked_for_questions = True
                    _reset_question_timers(session)
                    closing_audio = text_to_speech(closing, f"closing_{uuid.uuid4().hex}.mp3")
                    return {
                        "transcript": transcript,
                        "completed": False,
                        "next_question": closing,
                        "audio_url": closing_audio,
                        "question_number": session.current_question_number,
                        "max_questions": session.max_questions,  # Total questions including pre-closing and closing
                        "continuous": True
                    }
                
                # Normal flow: ask the closing question
                closing = generate_question(session, "closing", last_answer_text=transcript)
                session.add_interviewer_message(closing)
                session.current_question_number += 1
                session.awaiting_answer = True
                session.last_active_question_text = closing
                session.asked_for_questions = True
                _reset_question_timers(session)
                closing_audio = text_to_speech(closing, f"closing_{uuid.uuid4().hex}.mp3")
                return {
                    "transcript": transcript,
                    "completed": False,
                    "next_question": closing,
                    "audio_url": closing_audio,
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,  # Total questions including pre-closing and closing
                    "continuous": True
                }
            # Already in closing phase
            session.completed_at = time.time()
            session.is_completed = True
            return {
                "transcript": transcript,
                "completed": True,
                "message": "Interview completed. Thank you for your time!"
            }
        
        # Gate moving to next question until 5s of no new words after first voice
        if session.awaiting_answer and session.first_voice_at is not None:
            last_update = session.last_transcript_update_at or session.first_voice_at
            if (now_ts - last_update) < 5.0:
                return {
                    "transcript": transcript,
                    "completed": False,
                    "acknowledge": True,
                    "message": "Listening...",
                    "question_number": session.current_question_number,
                    "max_questions": session.max_questions,
                    "continuous": True
                }

        # Generate next question
        # Check if we've reached the limit for main questions (excluding pre-closing and closing)
        # The introduction question is question 1, so we need to account for that
        main_questions_after_intro = max(0, session.max_questions - 3)  # Questions after intro, before pre-closing
        main_questions_limit = 1 + main_questions_after_intro  # Total main questions including intro
        
        if session.current_question_number >= main_questions_limit:
            # We've reached the limit, should have been handled above
            # This shouldn't happen, but just in case, return error
            print(f"⚠️ WARNING: Reached main questions limit ({main_questions_limit}) but didn't trigger pre-closing")
            print(f"   Current: {session.current_question_number}, Max: {session.max_questions}")
            return {
                "transcript": transcript,
                "completed": False,
                "error": "Question limit reached"
            }
        
        # Decide whether to ask follow-up or regular question based on:
        # 1. Answer quality (broad/vague)
        # 2. JD context match
        # 3. Ratio constraint (30% follow-up, 70% regular)
        should_follow_up = should_ask_follow_up(session, transcript)
        
        if should_follow_up:
            next_question = generate_question(session, "follow_up", last_answer_text=transcript)
            session.follow_up_questions_count += 1
            print(f"🤖 Generating follow-up question {session.current_question_number + 1}/{session.max_questions} based on answer snippet:", (transcript[:120] + ('...' if len(transcript) > 120 else '')))
            total_asked = session.regular_questions_count + session.follow_up_questions_count
            if total_asked > 0:
                ratio = (session.follow_up_questions_count / total_asked) * 100
                print(f"📊 Follow-up ratio: {session.follow_up_questions_count} follow-ups / {total_asked} total = {ratio:.1f}%")
        else:
            next_question = generate_question(session, "regular", last_answer_text=transcript)
            session.regular_questions_count += 1
            print(f"🤖 Generating regular question {session.current_question_number + 1}/{session.max_questions} based on answer snippet:", (transcript[:120] + ('...' if len(transcript) > 120 else '')))
            total_asked = session.regular_questions_count + session.follow_up_questions_count
            if total_asked > 0:
                ratio = (session.follow_up_questions_count / total_asked) * 100
                print(f"📊 Follow-up ratio: {session.follow_up_questions_count} follow-ups / {total_asked} total = {ratio:.1f}%")
        
        session.add_interviewer_message(next_question)
        session.current_question_number += 1
        session.awaiting_answer = True
        session.last_active_question_text = next_question
        
        print(f"📊 Question progress: {session.current_question_number}/{session.max_questions} (main limit: {main_questions_limit})")
        
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
        print(f"❌ Error in /upload_answer: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def repeat_question(session_id: str) -> Dict:
    """Exact copy from app.py /repeat endpoint"""
    try:
        if not session_id or session_id not in sessions:
            return {"error": "Invalid session ID"}
        
        session = sessions[session_id]
        last_q = get_last_strict_question(session)
        
        if not last_q:
            return {"error": "No previous question found"}

        audio_url = text_to_speech(last_q, f"repeat_{uuid.uuid4().hex}.mp3")
        _reset_question_timers(session)
        return {
            "next_question": last_q,
            "audio_url": audio_url,
            "question_number": session.current_question_number,
            "max_questions": session.max_questions
        }
    except Exception as e:
        print(f"❌ Error in /repeat: {e}")
        return {"error": str(e)}

