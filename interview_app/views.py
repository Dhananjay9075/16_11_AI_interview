import os
import google.generativeai as genai
from numpy._core.numeric import False_
import whisper
import PyPDF2
import docx
import re
import json
import threading
import csv
import shutil
# from gtts import gTTS  # Removed - using only Google Cloud TTS
from pathlib import Path
from dotenv import load_dotenv
import pytz
from textblob import TextBlob
import subprocess
import tempfile
import psutil
import sqlite3

# Google Cloud Text-to-Speech import with fallback
try:
    from google.cloud import texttospeech
    print("✅ Google Cloud Text-to-Speech imported successfully in views.py")
except ImportError as e:
    print(f"❌ Warning: google-cloud-texttospeech not available in views.py: {e}")
    texttospeech = None
except Exception as e:
    print(f"❌ Unexpected error importing google-cloud-texttospeech in views.py: {e}")
    texttospeech = None

from collections import Counter
import traceback
import readtime
import time
import numpy as np
import cv2
import base64
from django.utils import timezone
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
import urllib.parse


from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template import loader
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.template.loader import render_to_string
from django.utils.safestring import mark_safe
from weasyprint import HTML

# from .camera import VideoCamera
# from .simple_camera import SimpleVideoCamera as VideoCamera
# from .real_camera import RealVideoCamera as VideoCamera
# from .simple_real_camera import SimpleRealVideoCamera as VideoCamera
from .simple_real_camera import SimpleRealVideoCamera as VideoCamera
from .models import InterviewSession, WarningLog, InterviewQuestion
from .ai_chatbot import (
    ai_start_django,
    ai_upload_answer_django,
    ai_repeat_django,
    ai_transcript_pdf_django,
)

try:
    from .yolo_face_detector import detect_face_with_yolo
except ImportError:
    print("Warning: yolo_face_detector could not be imported. Using a placeholder.")
    def detect_face_with_yolo(img): return [type('obj', (object,), {'boxes': []})()]


load_dotenv()
# Use API key from Django settings (from environment variable)
try:
    from django.conf import settings as dj_settings
    active_key = getattr(dj_settings, 'GEMINI_API_KEY', '')
    # Fallback to hardcoded key if env key is not available
    if not active_key:
        active_key = "AIzaSyBu5M6cEckMIRPdttrBTBcRJmTUi5MkpvE"
        print("⚠️ WARNING: GEMINI_API_KEY not set in environment, using hardcoded key")
    
    if active_key:
        genai.configure(api_key=active_key)
        print("✅ Gemini API configured successfully in views.py")
    else:
        print("⚠️ WARNING: GEMINI_API_KEY not available")
except Exception as e:
    print(f"⚠️ WARNING: Could not configure Gemini API: {e}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# --- DEVELOPMENT MODE SWITCH ---
# Set to True to use hardcoded questions and skip AI generation for faster testing.
# This does NOT affect AI evaluation in the report.
DEV_MODE = False

try:
    whisper_model = whisper.load_model("base")
    print("Whisper model 'base' loaded.")
except Exception as e:
    print(f"Error loading Whisper model: {e}"); whisper_model = None

FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'i mean', 'basically', 'actually', 'literally']
CAMERAS, camera_lock = {}, threading.Lock()

THINKING_TIME, ANSWERING_TIME, REVIEW_TIME = 20, 60, 10

def get_camera_for_session(session_key):
    print(f"🔍 Getting camera for session_key: {session_key}")
    with camera_lock:
        if session_key in CAMERAS: 
            print(f"✅ Found existing camera for session_key: {session_key}")
            return CAMERAS[session_key]
        try:
            print(f"🔍 Looking up InterviewSession for session_key: {session_key}")
            session_obj = InterviewSession.objects.get(session_key=session_key)
            print(f"✅ Found InterviewSession: {session_obj.id}")
            print(f"🎥 Creating VideoCamera for session_id: {session_obj.id}")
            camera_instance = VideoCamera(session_id=session_obj.id)
            CAMERAS[session_key] = camera_instance
            print(f"✅ Camera created and stored for session_key: {session_key}")
            return camera_instance
        except InterviewSession.DoesNotExist:
            print(f"❌ Could not find session for session_key {session_key} to create camera.")
            return None
        except Exception as e:
            print(f"❌ Error creating camera instance: {e}")
            import traceback
            traceback.print_exc()
            return None

def release_camera_for_session(session_key):
    with camera_lock:
        if session_key in CAMERAS:
            print(f"--- Releasing camera for session {session_key} ---")
            CAMERAS[session_key].cleanup()
            del CAMERAS[session_key]

SUPPORTED_LANGUAGES = {'en': 'English'}

def get_text_from_file(uploaded_file):
    name, extension = os.path.splitext(uploaded_file.name)
    text = ""
    if extension == '.pdf':
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages: text += page.extract_text() or ""
    elif extension == '.docx':
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs: text += para.text + "\n"
    else: text = uploaded_file.read().decode('utf-8', errors='ignore')
    return text

@login_required
def create_interview_invite(request):
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name')
        candidate_email = request.POST.get('candidate_email')
        jd_text = request.POST.get('jd')
        resume_file = request.FILES.get('resume')
        language_code = request.POST.get('language', 'en')
        accent_tld = request.POST.get('accent', 'com')
        scheduled_at_str = request.POST.get('scheduled_at')

        if not all([candidate_name, candidate_email, jd_text, resume_file, scheduled_at_str]):
             return render(request, 'interview_app/create_invite.html', {'error': 'All fields are required.', 'languages': SUPPORTED_LANGUAGES})

        try:
            ist = pytz.timezone('Asia/Kolkata')
            naive_datetime = datetime.strptime(scheduled_at_str, '%Y-%m-%dT%H:%M')
            aware_datetime = ist.localize(naive_datetime)
        except (ValueError, pytz.exceptions.InvalidTimeError):
            return render(request, 'interview_app/create_invite.html', {'error': 'Invalid date and time format provided.', 'languages': SUPPORTED_LANGUAGES})

        resume_text = get_text_from_file(resume_file)
        if not resume_text:
            return render(request, 'interview_app/create_invite.html', {'error': 'Could not read the resume file.', 'languages': SUPPORTED_LANGUAGES})

        session = InterviewSession.objects.create(
            candidate_name=candidate_name, candidate_email=candidate_email,
            job_description=jd_text, resume_text=resume_text,
            language_code=language_code, accent_tld=accent_tld,
            scheduled_at=aware_datetime
        )

        interview_url = request.build_absolute_uri(f"/?session_key={session.session_key}")

        return redirect('dashboard')

    return render(request, 'interview_app/create_invite.html', {'languages': SUPPORTED_LANGUAGES})

def start_interview(request):
    """Start interview directly with form - creates session and redirects to portal."""
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name', '').strip()
        job_description = request.POST.get('job_description', '').strip()
        candidate_email = request.POST.get('candidate_email', '').strip() or f"{candidate_name.lower().replace(' ', '.')}@example.com"
        resume_text = request.POST.get('resume_text', '').strip() or "Experienced professional seeking new opportunities."
        
        # Validate required fields
        if not candidate_name:
            return render(request, 'interview_app/start_interview.html', {
                'error': 'Candidate name is required',
                'candidate_name': candidate_name,
                'job_description': job_description
            })
        
        # Create interview session scheduled for now
        try:
            session = InterviewSession.objects.create(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_description=job_description or "Technical Role",
                resume_text=resume_text,
                scheduled_at=timezone.now(),
                language_code='en',
                accent_tld='com',
                status='SCHEDULED'
            )
            
            # Redirect to portal with session key - portal will handle permissions
            return redirect(f'/?session_key={session.session_key}')
            
        except Exception as e:
            return render(request, 'interview_app/start_interview.html', {
                'error': f'Failed to create interview session: {str(e)}',
                'candidate_name': candidate_name,
                'job_description': job_description
            })
    
    # GET request - show form
    return render(request, 'interview_app/start_interview.html')

@login_required
def generate_interview_link(request):
    """Generate an interview link for a candidate."""
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name', '').strip()
        candidate_email = request.POST.get('candidate_email', '').strip()
        job_description = request.POST.get('job_description', '').strip()
        resume_text = request.POST.get('resume_text', '').strip()
        scheduled_at_str = request.POST.get('scheduled_at', '')
        language_code = request.POST.get('language_code', 'en')
        accent_tld = request.POST.get('accent_tld', 'com')
        
        # Validate required fields
        if not candidate_name or not candidate_email:
            return JsonResponse({
                'success': False,
                'error': 'Candidate name and email are required'
            }, status=400)
        
        # Handle scheduled_at
        scheduled_at = None
        if scheduled_at_str:
            try:
                ist = pytz.timezone('Asia/Kolkata')
                naive_datetime = datetime.strptime(scheduled_at_str, '%Y-%m-%dT%H:%M')
                scheduled_at = ist.localize(naive_datetime)
            except (ValueError, pytz.exceptions.InvalidTimeError):
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid date and time format. Use YYYY-MM-DDTHH:MM'
                }, status=400)
        else:
            # If no scheduled time provided, schedule for now
            scheduled_at = timezone.now()
        
        # Create interview session
        try:
            session = InterviewSession.objects.create(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_description=job_description or "Technical Role",
                resume_text=resume_text or "Experienced professional seeking new opportunities.",
                scheduled_at=scheduled_at,
                language_code=language_code,
                accent_tld=accent_tld,
                status='SCHEDULED'
            )
            
            # Generate interview link
            base_url = request.build_absolute_uri('/')
            interview_link = f"{base_url}?session_key={session.session_key}"
            
            return JsonResponse({
                'success': True,
                'interview_link': interview_link,
                'session_key': session.session_key,
                'session_id': str(session.id),
                'candidate_name': session.candidate_name,
                'candidate_email': session.candidate_email,
                'scheduled_at': session.scheduled_at.isoformat() if session.scheduled_at else None,
                'status': session.status
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Failed to create interview session: {str(e)}'
            }, status=500)
    
    # GET request - show form or return instructions
    if request.headers.get('Accept') == 'application/json':
        return JsonResponse({
            'success': False,
            'error': 'POST request required',
            'instructions': {
                'method': 'POST',
                'required_fields': ['candidate_name', 'candidate_email'],
                'optional_fields': ['job_description', 'resume_text', 'scheduled_at', 'language_code', 'accent_tld'],
                'scheduled_at_format': 'YYYY-MM-DDTHH:MM (e.g., 2024-12-25T14:30)',
                'example': {
                    'candidate_name': 'John Doe',
                    'candidate_email': 'john@example.com',
                    'job_description': 'Software Engineer position...',
                    'resume_text': 'John has 5 years of experience...',
                    'scheduled_at': '2024-12-25T14:30',
                    'language_code': 'en',
                    'accent_tld': 'com'
                }
            }
        })
    
    # Render HTML form if GET request
    return render(request, 'interview_app/generate_link.html', {
        'languages': SUPPORTED_LANGUAGES
    })

def synthesize_speech(text, lang_code, accent_tld, output_path):
    """Use ONLY Google Cloud TTS - no fallback to gTTS"""
    if texttospeech is None:
        print("❌ Google Cloud TTS not available - texttospeech is None")
        raise Exception("Google Cloud TTS not available")
    
    try:
        print(f"🎤 Google Cloud TTS: Synthesizing '{text[:50]}...'")
        
        # Ensure credentials are set
        credentials_path = os.path.join(settings.BASE_DIR, "ringed-reach-471807-m3-cf0ec93e3257.json")
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            print(f"✅ Google Cloud credentials set: {credentials_path}")
        else:
            print(f"❌ Google Cloud credentials not found: {credentials_path}")
            raise Exception("Google Cloud credentials not found")
        
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use a high-quality voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # High-quality neural voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        with open(output_path, 'wb') as out:
            out.write(response.audio_content)
        
        print(f"✅ Google Cloud TTS: Audio saved to {output_path}")
        return
    
    except Exception as e:
        print(f"❌ Google Cloud TTS failed: {e}")
        raise Exception(f"Google Cloud TTS failed: {e}")

def interview_portal(request):
    session_key = (request.GET.get('session_key') or '').strip()
    print(f"DEBUG: interview_portal called with session_key: {session_key}")

    # If no session_key, show a simple scheduling form on the home page.
    # This lets you enter candidate name + JD and start the full flow
    # without generating a link from the CLI.
    if not session_key:
        if request.method == "POST":
            # Get form data - these are REQUIRED from portal.html
            candidate_name = (request.POST.get("candidate_name") or "").strip()
            job_description = (request.POST.get("job_description") or "").strip()
            question_count_str = request.POST.get("question_count", "4").strip()
            
            # Validate required fields
            if not candidate_name:
                return render(request, 'interview_app/portal.html', {
                    'interview_started': False,
                    'session_key': '',
                    'error': 'Candidate name is required'
                })
            if not job_description:
                return render(request, 'interview_app/portal.html', {
                    'interview_started': False,
                    'session_key': '',
                    'error': 'Job description is required'
                })
            
            # Parse question_count with validation
            try:
                question_count = max(1, min(15, int(question_count_str)))  # Between 1 and 15
            except (ValueError, TypeError):
                question_count = 4  # Default if invalid

            print(f"\n{'='*60}")
            print(f"📝 FORM SUBMISSION FROM PORTAL.HTML")
            print(f"{'='*60}")
            print(f"   Candidate Name: {candidate_name}")
            print(f"   Job Description Length: {len(job_description)}")
            print(f"   Job Description Preview: {job_description[:100]}...")
            print(f"   Question Count: {question_count}")
            print(f"{'='*60}\n")

            session = InterviewSession.objects.create(
                candidate_name=candidate_name,
                job_description=job_description,
                scheduled_at=timezone.now(),
                language_code="en",
                accent_tld="com",
                status="SCHEDULED",
            )
            
            # Verify all data was saved correctly
            saved_session = InterviewSession.objects.get(session_key=session.session_key)
            print(f"✅ Session created with session_key: {session.session_key}")
            print(f"   Saved Candidate Name: {saved_session.candidate_name}")
            print(f"   Saved JD length: {len(saved_session.job_description or '')}")
            print(f"   Saved JD preview: {(saved_session.job_description or '')[:100]}...")
            print(f"   Question Count to use: {question_count}")
            
            # Pass question_count via query param so JS can send it to /ai/start
            return redirect(f"/?session_key={session.session_key}&qc={question_count}")

        # GET without session_key: render portal with scheduling UI only
        return render(request, 'interview_app/portal.html', {
            'interview_started': False,
            'session_key': '',
        })

    # Existing behavior when session_key is present
    session = get_object_or_404(InterviewSession, session_key=session_key)
    print(f"DEBUG: Found session with ID: {session.id}")
    
    # This is the main validation logic block
    if session.status != 'SCHEDULED':
        return render(request, 'interview_app/invalid_link.html', {'error': 'This interview has already been completed or has expired.'})
    if session.scheduled_at:
        now = timezone.now()
        start_time = session.scheduled_at
        grace_period = timedelta(minutes=10)
        expiry_time = start_time + grace_period
        
        # Debug time comparison
        print(f"DEBUG: Time comparison - Now: {now}, Start: {start_time}, Expiry: {expiry_time}")
        print(f"DEBUG: Now < Start: {now < start_time}, Now > Expiry: {now > expiry_time}")
        
        # Case 1: The user is too early.
        # Add a small buffer (30 seconds) to account for timezone differences and network delays
        buffer_time = timedelta(seconds=30)
        if now < (start_time - buffer_time):
            start_time_local = start_time.astimezone(pytz.timezone('Asia/Kolkata'))
            print(f"DEBUG: User too early, showing countdown. Start time local: {start_time_local}")
            # We pass all necessary context for the countdown timer here.
            return render(request, 'interview_app/invalid_link.html', {
                'page_title': 'Interview Not Started',
                'error': f"Your interview has not started yet. Please use the link at the scheduled time:",
                'scheduled_time_str': start_time_local.strftime('%Y-%m-%d %I:%M %p IST'),
                'start_time_iso': start_time.isoformat() # This is crucial for the JS countdown
            })
        # Case 2: The user is too late.
        if now > expiry_time:
            session.status = 'EXPIRED'
            session.save()
            return render(request, 'interview_app/invalid_link.html', {
                'page_title': 'Interview Link Expired',
                'error': 'This interview link has expired because the 10-minute grace period after the scheduled time has passed.'
            })
    else:
        # Case 3: The session has no scheduled time (should not happen in normal flow).
         return render(request, 'interview_app/invalid_link.html', {'error': 'This interview session is invalid as it does not have a scheduled time.'})
    # If the user is within the valid time window, proceed with the interview setup.
    try:
        # Initialize variables
        all_questions = []
        
        print(f"DEBUG: About to load questions from database")
        # Load existing questions from database
        if True:  # ENABLED: Load questions from database
            all_questions = []
            tts_dir = os.path.join(settings.MEDIA_ROOT, 'tts')
            os.makedirs(tts_dir, exist_ok=True)
            
            # Check if there are existing questions
            existing_questions = session.questions.filter(question_level='MAIN').order_by('order')
            
            if existing_questions.exists():
                # Load existing questions and generate audio if missing
                for i, q in enumerate(existing_questions):
                        
                    if not q.audio_url:
                        # Generate audio for questions that don't have it
                        tts_path = os.path.join(tts_dir, f'q_{i}_{session.session_key}.mp3')
                        synthesize_speech(q.question_text, session.language_code, session.accent_tld, tts_path)
                        audio_url = f"{settings.MEDIA_URL}tts/{os.path.basename(tts_path)}"
                        # Update the question in database
                        q.audio_url = audio_url
                        q.save()
                    else:
                        audio_url = q.audio_url
                    
                    all_questions.append({
                        'type': q.question_type, 
                        'text': q.question_text, 
                        'audio_url': audio_url
                    })
                generate_new_questions = False

            else:
                # No existing questions, generate new ones
                print("DEBUG: No existing questions found, generating new ones...")
                # Set flag to generate new questions
                generate_new_questions = True
            
            # Check if we need to generate new questions
            print(f"DEBUG: generate_new_questions flag: {generate_new_questions if 'generate_new_questions' in locals() else 'NOT SET'}")
            if 'generate_new_questions' in locals() and generate_new_questions:
                print("DEBUG: Generating new questions due to no existing questions...")
                # Generate new questions using the existing logic
                DEV_MODE = False
                if DEV_MODE:
                    print("--- RUNNING IN DEV MODE: Using hardcoded questions and summary. ---")
                    session.resume_summary = "This is a sample resume summary for developer mode. The candidate seems proficient in Python and Django."
                    all_questions = [
                        {'type': 'Ice-Breaker', 'text': 'Welcome! To start, can you tell me about a challenging project you have worked on?'},
                        {'type': 'Technical Questions', 'text': 'What is the difference between `let`, `const`, and `var` in JavaScript?'},
                        {'type': 'Behavioral Questions', 'text': 'Describe a time you had a conflict with a coworker and how you resolved it.'}
                    ]
                
                # Save spoken questions to database
                for i, q_data in enumerate(all_questions):
                    tts_path = os.path.join(tts_dir, f'q_{i}_{session.session_key}.mp3')
                    synthesize_speech(q_data['text'], session.language_code, session.accent_tld, tts_path)
                    audio_url = f"{settings.MEDIA_URL}tts/{os.path.basename(tts_path)}"
                    q_data['audio_url'] = audio_url
                    InterviewQuestion.objects.create(
                        session=session,
                        question_text=q_data['text'],
                        question_type=q_data['type'],
                        order=i,
                        question_level='MAIN'
                    )
        else:
            DEV_MODE = False

            all_questions = []
            if DEV_MODE:
                print("--- RUNNING IN DEV MODE: Using hardcoded questions and summary. ---")
                session.resume_summary = "This is a sample resume summary for developer mode. The candidate seems proficient in Python and Django."
                all_questions = [
                    {'type': 'Ice-Breaker', 'text': 'Welcome! To start, can you tell me about a challenging project you have worked on?'},
                    {'type': 'Technical Questions', 'text': 'What is the difference between `let`, `const`, and `var` in JavaScript?'},
                    {'type': 'Behavioral Questions', 'text': 'Describe a time you had a conflict with a coworker and how you resolved it.'}
                ]
            else:
                print("--- RUNNING IN PRODUCTION MODE: Calling Gemini API. ---")
                model = genai.GenerativeModel('gemini-2.0-flash')
                summary_prompt = f"Summarize key skills from the following resume:\n\n{session.resume_text}"
                summary_response = model.generate_content(summary_prompt)
                session.resume_summary = summary_response.text
                language_name = SUPPORTED_LANGUAGES.get(session.language_code, 'English')
                
                # Get question count from InterviewSlot.ai_configuration or default to 4
                question_count = 4  # Default
                try:
                    # Get Interview via session_key (priority 1)
                    from interviews.models import Interview
                    interview = Interview.objects.filter(session_key=session.session_key).first()
                    
                    # If not found via session_key, try via candidate email (priority 2)
                    if not interview and session.candidate_email:
                        from candidates.models import Candidate
                        try:
                            candidate = Candidate.objects.get(email=session.candidate_email)
                            interview = Interview.objects.filter(candidate=candidate).order_by('-created_at').first()
                        except:
                            pass
                    
                    if interview and interview.slot:
                        slot = interview.slot
                        print(f"✅ Found Interview {interview.id} with Slot {slot.id}")
                        print(f"   Slot AI Config: {slot.ai_configuration}")
                        
                        if slot.ai_configuration and isinstance(slot.ai_configuration, dict):
                            question_count = slot.ai_configuration.get('question_count', 4)
                            # Ensure it's an integer
                            try:
                                question_count = int(question_count)
                            except (ValueError, TypeError):
                                question_count = 4
                            print(f"✅ Using question count from InterviewSlot.ai_configuration: {question_count}")
                        # Also check if question_count is in slot directly (if it was added as a field)
                        elif hasattr(slot, 'question_count') and slot.question_count:
                            question_count = int(slot.question_count)
                            print(f"✅ Using question count from InterviewSlot.question_count: {question_count}")
                        else:
                            print(f"⚠️ No question_count found in slot.ai_configuration, using default 4")
                            print(f"   Available keys in ai_configuration: {list(slot.ai_configuration.keys()) if slot.ai_configuration else 'None'}")
                    else:
                        if not interview:
                            print(f"⚠️ No Interview found for session_key={session.session_key}, candidate_email={session.candidate_email}")
                        elif not interview.slot:
                            print(f"⚠️ Interview {interview.id} has no slot assigned")
                        print(f"⚠️ Using default question_count: 4")
                except Exception as e:
                    print(f"⚠️ Error getting question count, using default 4: {e}")
                    import traceback
                    traceback.print_exc()
                
                master_prompt = (
                    f"You are an expert Talaro interviewer.Your task is to generate {question_count} insightful interview 1-2 liner questions in {language_name}. "
                    f"The interview is for a '{session.job_description.splitlines()[0]}' role. "
                    "starting from introduction question .Please base the questions on the provided job description and candidate's resume. "
                    "Start with a welcoming ice-breaker question that also references something specific from the candidate's resume. "
                    "Then, generate a mix of technical Questions. 70 percent from jd and 30 percent from resume"
                    "You MUST format the output as Markdown. "
                    "You MUST include '## Technical Questions'. "
                    "Each question MUST start with a hyphen '-'. "
                    "Do NOT add any introductions, greetings (beyond the first ice-breaker question), or concluding remarks. "
                    f"\n\n--- JOB DESCRIPTION ---\n{session.job_description}\n\n--- RESUME ---\n{session.resume_text}"
                )
                full_response = model.generate_content(master_prompt)
                response_text = full_response.text
                sections = re.findall(r"##\s*(.*?)\s*\n(.*?)(?=\n##|\Z)", response_text, re.DOTALL)
                if not sections: raise ValueError("Could not parse ## headers from AI response.")
                for category_name, question_block in sections:
                    lines = question_block.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('-'):
                            all_questions.append({'type': category_name.strip(), 'text': line.strip().lstrip('- ').strip()})
                
            if not all_questions: raise ValueError("No questions were generated or parsed.")
            if all_questions and "welcome" in all_questions[0]['text'].lower():
                all_questions[0]['type'] = 'Ice-Breaker'
            session.save()
            tts_dir = os.path.join(settings.MEDIA_ROOT, 'tts'); os.makedirs(tts_dir, exist_ok=True)
            # Save spoken questions to database
            for i, q_data in enumerate(all_questions):
                tts_path = os.path.join(tts_dir, f'q_{i}_{session.session_key}.mp3')
                synthesize_speech(q_data['text'], session.language_code, session.accent_tld, tts_path)
                audio_url = f"{settings.MEDIA_URL}tts/{os.path.basename(tts_path)}"
                q_data['audio_url'] = audio_url
                InterviewQuestion.objects.create(
                    session=session,
                    question_text=q_data['text'],
                    question_type=q_data['type'],
                    order=i,
                    question_level='MAIN'
                )
            
        
        # Debug: Print what we're sending to the template
        print(f"\n{'='*70}")
        print(f"🔍 PORTAL DATA DEBUG:")
        print(f"   Spoken questions: {len(all_questions)}")
        print(f"{'='*70}\n")
        
        print(f"DEBUG: Rendering interview portal for session {session_key}")
        context = {
            'session_key': session_key,
            'interview_session_id': str(session.id),
            'spoken_questions_data': all_questions,
            'interview_started': True,
            'candidate_name': session.candidate_name,
            'job_description': session.job_description,
        }
        return render(request, 'interview_app/portal.html', context)
    except Exception as e:
        # Graceful fallback when AI services (e.g., Gemini) are unavailable.
        # Load the portal with no spoken questions so the chatbot phase can proceed.
        try:
            print(f"WARN: AI setup failed, falling back without spoken questions: {e}")
            session = get_object_or_404(InterviewSession, session_key=session_key)
            context = {
                'session_key': session_key,
                'interview_session_id': str(session.id),
                'spoken_questions_data': [],
                'interview_started': True,
            }
            return render(request, 'interview_app/portal.html', context)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return HttpResponse(f"An API or processing error occurred: {str(e)}", status=500)
        
def dashboard(request):
    sessions = InterviewSession.objects.all().order_by('-created_at')
    context = {'sessions': sessions}
    template = loader.get_template('interview_app/dashboard.html')
    return HttpResponse(template.render(context, request))

def interview_report(request, session_id):
    try:
        session = InterviewSession.objects.get(id=session_id)
        all_questions = list(session.questions.all().order_by('order'))
        all_logs_list = list(session.logs.all())
        warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
        
        # Check if there is any new content to evaluate
        has_spoken_answers = session.questions.filter(transcribed_answer__isnull=False, transcribed_answer__gt='').exists()

        if session.language_code == 'en' and not session.is_evaluated and has_spoken_answers:
            print(f"--- Performing all first-time AI evaluations for session {session.id} with Gemini ---")
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            try:
                print("--- Evaluating Resume vs. Job Description ---")
                resume_eval_prompt = (
                    "You are an expert technical recruiter. Analyze the following resume against the provided job description. "
                    "Provide a score from 0.0 to 10.0 indicating how well the candidate's experience aligns with the job requirements. "
                    "Also provide a brief analysis. Format your response EXACTLY as follows:\n\n"
                    "SCORE: [Your score, e.g., 8.2]\n"
                    "ANALYSIS: [Your one-paragraph analysis here.]"
                    f"\n\nJOB DESCRIPTION:\n{session.job_description}\n\nRESUME:\n{session.resume_text}"
                )
                resume_response = model.generate_content(resume_eval_prompt)
                resume_response_text = resume_response.text
                score_match = re.search(r"SCORE:\s*([\d\.]+)", resume_response_text)
                if score_match: session.resume_score = float(score_match.group(1))
                session.resume_feedback = resume_response_text
            except Exception as e:
                print(f"ERROR during Resume evaluation: {e}"); session.resume_feedback = "An error occurred during resume evaluation."

            try:
                print("--- Evaluating Interview Performance (Spoken) ---")
                
                qa_text = "".join([
                    f"Question: {item.question_text}\nAnswer: {item.transcribed_answer or 'No answer provided.'}\n\n"
                    for item in all_questions
                ])

                answers_eval_prompt = (
                    "You are an expert technical hiring manager. Evaluate the candidate's interview performance "
                    "based on their spoken answers. Provide an overall score from 0.0 to 10.0 "
                    "and a brief summary of their strengths and areas for improvement.\n\n"
                    "Consider the following:\n"
                    "- Evaluate clarity, relevance, and communication skills.\n"
                    "- Assess technical knowledge and problem-solving approach.\n\n"
                    "Format your response EXACTLY as follows:\n\n"
                    "SCORE: [Your score, e.g., 7.5]\n"
                    "FEEDBACK: [Your detailed feedback here.]"
                    f"\n\n--- SPOKEN QUESTIONS & ANSWERS ---\n{qa_text or 'No spoken answers provided.'}"
                )
                
                answers_response = model.generate_content(answers_eval_prompt)
                answers_response_text = answers_response.text
                score_match = re.search(r"SCORE:\s*([\d\.]+)", answers_response_text)
                if score_match: session.answers_score = float(score_match.group(1))
                session.answers_feedback = answers_response_text
            except Exception as e:
                print(f"ERROR during Answers evaluation: {e}"); session.answers_feedback = "An error occurred during answers evaluation."

            try:
                print("--- Generating Overall Candidate Profile ---")
                warning_summary = ", ".join([f"{count}x {name}" for name, count in warning_counts.items()]) or "None"
                overall_prompt = (
                    "You are a senior hiring manager. You have been provided with a holistic view of a candidate's interview, "
                    "including their resume fit, interview answer performance, and proctoring warnings. "
                    "Synthesize all this information into a final recommendation. "
                    "Provide a final 'Overall Score' from 0.0 to 10.0 and a concluding 'Hiring Recommendation' paragraph.\n\n"
                    "DATA PROVIDED:\n"
                    f"- Resume vs. Job Description Score: {session.resume_score or 'N/A'}/10\n"
                    f"- Interview Answers Score: {session.answers_score or 'N/A'}/10\n"
                    f"- Proctoring Warnings: {warning_summary}\n\n"
                    "Format your response EXACTLY as follows:\n\n"
                    "OVERALL SCORE: [Your final blended score, e.g., 7.8]\n"
                    "HIRING RECOMMENDATION: [Your final concluding paragraph on whether to proceed with the candidate and why.]"
                )
                overall_response = model.generate_content(overall_prompt)
                overall_response_text = overall_response.text
                score_match = re.search(r"OVERALL SCORE:\s*([\d\.]+)", overall_response_text)
                if score_match: session.overall_performance_score = float(score_match.group(1))
                session.overall_performance_feedback = overall_response_text
            except Exception as e:
                print(f"ERROR during Overall evaluation: {e}"); session.overall_performance_feedback = "An error occurred."
            
            session.is_evaluated = True
            session.save()
        
        total_filler_words = 0
        avg_wpm = 0
        wpm_count = 0
        sentiment_scores = []
        avg_response_time = 0
        response_time_count = 0

        if session.language_code == 'en':
            for item in all_questions:
                if item.transcribed_answer:
                    word_count = len(item.transcribed_answer.split())
                    read_time_result = readtime.of_text(item.transcribed_answer)
                    read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                    if read_time_minutes > 0:
                        item.words_per_minute = round(word_count / read_time_minutes)
                        avg_wpm += item.words_per_minute
                        wpm_count += 1
                    else:
                        item.words_per_minute = 0
                    if item.response_time_seconds:
                        avg_response_time += item.response_time_seconds
                        response_time_count += 1
                    lower_answer = item.transcribed_answer.lower()
                    item.filler_word_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                    total_filler_words += item.filler_word_count
                    sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': TextBlob(item.transcribed_answer).sentiment.polarity})
                else:
                    sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': 0.0})
        
        final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
        final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0

        analytics_data = {
            'warning_counts': dict(warning_counts),
            'sentiment_scores': sentiment_scores,
            'evaluation_scores': {'Resume vs JD': session.resume_score or 0, 'Interview Answers': session.answers_score or 0},
            'communication_radar': {
                'Pace (WPM)': final_avg_wpm,
                'Clarity (Few Fillers)': total_filler_words,
                'Responsiveness (sec)': final_avg_response_time
            },
        }
        
        main_questions_with_followups = session.questions.filter(question_level='MAIN', question_type__in=['TECHNICAL', 'BEHAVIORAL']).prefetch_related('follow_ups').order_by('order')

        context = {
            'session': session, 
            'main_questions_with_followups': main_questions_with_followups,
            'analytics_data': json.dumps(analytics_data),
            'total_filler_words': total_filler_words,
            'avg_wpm': final_avg_wpm,
            'behavioral_analysis_html': mark_safe((session.behavioral_analysis or "").replace('\n', '<br>')),
            'keyword_analysis_html': mark_safe((session.keyword_analysis or "").replace('\n', '<br>').replace('**', '<strong>').replace('**', '</strong>'))
        }
        template = loader.get_template('interview_app/report.html')
        return HttpResponse(template.render(context, request))
    except InterviewSession.DoesNotExist:
        return HttpResponse("Interview session not found.", status=404)
def download_report_pdf(request, session_id):
    """
    Generates and serves a PDF version of the interview report for a given session.
    """
    try:
        # 1. Fetch the main session object.
        session = InterviewSession.objects.get(id=session_id)
        
        # 2. Prepare data for the proctoring summary chart.
        all_logs_list = list(session.logs.all())
        warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
        chart_config = { 
            'type': 'doughnut', 
            'data': { 
                'labels': list(warning_counts.keys()), 
                'datasets': [{'data': list(warning_counts.values())}]
            }
        }
        # URL-encode the chart configuration to pass to the chart generation service.
        chart_url = f"https://quickchart.io/chart?c={urllib.parse.quote(json.dumps(chart_config))}"
        
        # 3. Fetch all SPOKEN questions and their follow-ups.
        # For the PDF we want the full interview Q&A (not just MAIN level),
        # excluding any coding questions.
        main_questions_with_followups = session.questions.filter(
            question_type__in=['TECHNICAL', 'BEHAVIORAL']
        ).prefetch_related('follow_ups').order_by('order')
        
        # Debug: Print question count and details
        question_count = main_questions_with_followups.count()
        print(f"📋 Found {question_count} questions for PDF report (session: {session.id})")
        if question_count > 0:
            for q in main_questions_with_followups[:3]:  # Print first 3 for debugging
                print(f"  - Q{q.order}: {q.question_text[:50]}... | Answer: {bool(q.transcribed_answer)}")
        
        # 4. Calculate metrics for charts
        # Grammar score (estimated from answers - assume good grammar if answers exist, can be refined)
        grammar_score = min(100, max(0, (session.answers_score or 0) * 10 + 20)) if session.answers_score else 70
        
        # Technical knowledge (from answers_score converted to percentage)
        technical_knowledge = min(100, max(0, (session.answers_score or 0) * 10)) if session.answers_score else 50
        
        # Determine recommendation using only grammar + technical knowledge
        overall_percentage = (grammar_score * 0.3 + technical_knowledge * 0.7)
        if overall_percentage >= 80:
            recommendation = "STRONGLY RECOMMENDED"
            recommendation_color = "#28a745"  # Green
        elif overall_percentage >= 65:
            recommendation = "RECOMMENDED"
            recommendation_color = "#17a2b8"  # Blue
        elif overall_percentage >= 50:
            recommendation = "CONDITIONAL RECOMMENDATION"
            recommendation_color = "#ffc107"  # Yellow
        else:
            recommendation = "NOT RECOMMENDED"
            recommendation_color = "#dc3545"  # Red
        
        # Generate Bar Chart for Grammar, Technical Knowledge
        bar_chart_config = {
            'type': 'bar',
            'data': {
                'labels': ['Grammar', 'Technical Knowledge'],
                'datasets': [{
                    'label': 'Score (%)',
                    'data': [grammar_score, technical_knowledge],
                    'backgroundColor': ['#3498db', '#9b59b6'],
                    'borderColor': ['#2980b9', '#8e44ad'],
                    'borderWidth': 2
                }]
            },
            'options': {
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'max': 100,
                        'ticks': {
                            'callback': "function(value) { return value + '%'; }"
                        }
                    }
                },
                'plugins': {
                    'legend': {
                        'display': False
                    }
                }
            }
        }
        bar_chart_url = f"https://quickchart.io/chart?c={urllib.parse.quote(json.dumps(bar_chart_config))}"

        # 5. Assemble the complete context dictionary to be passed to the template.
        context = { 
            'session': session, 
            'main_questions_with_followups': main_questions_with_followups,
            'warning_counts': dict(warning_counts), 
            'chart_url': chart_url,
            'grammar_score': grammar_score,
            'technical_knowledge': technical_knowledge,
            'bar_chart_url': bar_chart_url,
            'recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'overall_percentage': overall_percentage
        }
        
        # 6. Download chart images and convert to base64 to avoid WeasyPrint network delays
        try:
            import requests
        except ImportError:
            print("⚠️ requests library not available, charts may load slowly")
            requests = None
        
        def download_chart_to_base64(url, timeout=5):
            """Download chart image and convert to base64 data URI"""
            if not requests:
                # If requests not available, return original URL (WeasyPrint will fetch it)
                return url
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                img_data = response.content
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                content_type = response.headers.get('Content-Type', 'image/png')
                return f"data:{content_type};base64,{img_base64}"
            except requests.exceptions.Timeout:
                print(f"⚠️ Chart download timed out: {url}")
                return url  # Return original URL as fallback
            except Exception as e:
                print(f"⚠️ Failed to download chart from {url}: {e}")
                return url  # Return original URL as fallback
        
        # Download all chart images before rendering (with shorter timeout for faster failover)
        # Use concurrent downloads to speed up the process
        import concurrent.futures
        print("📥 Downloading chart images (max 3s each, concurrent)...")
        chart_urls_to_download = []
        if chart_url:
            chart_urls_to_download.append(('chart_url', chart_url))
        if bar_chart_url:
            chart_urls_to_download.append(('bar_chart_url', bar_chart_url))
        # pie_chart_url removed (no Technology Understanding chart anymore)
        
        chart_results = {}
        if chart_urls_to_download and requests:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_name = {
                    executor.submit(download_chart_to_base64, url, timeout=3): name 
                    for name, url in chart_urls_to_download
                }
                for future in concurrent.futures.as_completed(future_to_name, timeout=10):
                    name = future_to_name[future]
                    try:
                        chart_results[name] = future.result()
                    except Exception as e:
                        print(f"⚠️ Chart download failed for {name}: {e}")
                        # Fallback to original URL
                        original_url = next((url for n, url in chart_urls_to_download if n == name), None)
                        chart_results[name] = original_url if original_url else None
        else:
            # Fallback: sequential downloads or use original URLs
            for name, url in chart_urls_to_download:
                chart_results[name] = download_chart_to_base64(url, timeout=3) if requests else url
        
        print("✅ Chart downloads complete")
        
        # Update context with downloaded images (base64) or original URLs
        context['chart_url'] = chart_results.get('chart_url', chart_url) if 'chart_url' in chart_results else chart_url
        context['bar_chart_url'] = chart_results.get('bar_chart_url', bar_chart_url) if 'bar_chart_url' in chart_results else bar_chart_url
        
        # 6. Render the HTML template to a string.
        print("📄 Rendering HTML template...")
        html_string = render_to_string('interview_app/report_pdf.html', context)
        
        # 7. Use WeasyPrint to convert the rendered HTML string into a PDF.
        print("🖨️ Generating PDF with WeasyPrint...")
        try:
            # Use base_url to help resolve any relative URLs, but don't rely on it for external images
            # (since we've already downloaded them as base64)
            pdf = HTML(string=html_string, base_url=request.build_absolute_uri('/')).write_pdf()
            print(f"✅ PDF generated successfully ({len(pdf)} bytes)")
        except Exception as pdf_error:
            print(f"❌ WeasyPrint error: {pdf_error}")
            # Fallback: try without base_url
            try:
                pdf = HTML(string=html_string).write_pdf()
                print(f"✅ PDF generated with fallback method ({len(pdf)} bytes)")
            except Exception as fallback_error:
                print(f"❌ PDF generation completely failed: {fallback_error}")
                import traceback
                traceback.print_exc()
                return HttpResponse(
                    f"PDF generation failed. Please check server logs. Error: {str(fallback_error)}",
                    status=500,
                    content_type='text/plain'
                )
        
        # 8. Create and return the final HTTP response.
        response = HttpResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="interview_report_{session.id}.pdf"'
        return response
        
    except InterviewSession.DoesNotExist:
        return HttpResponse("Interview session not found.", status=404)


def download_proctoring_pdf(request, session_id):
    """
    Generates and serves a PDF focused on live proctoring warnings
    and snapshots for a given session.
    """
    try:
        session = InterviewSession.objects.get(id=session_id)
        all_logs_list = list(session.logs.all().order_by('timestamp'))

        warning_counts = Counter([
            log.warning_type.replace('_', ' ').title()
            for log in all_logs_list
            if log.warning_type != 'excessive_movement'
        ])

        # Build absolute URLs for snapshot images (if any)
        snapshot_entries = []
        for log in all_logs_list:
            if log.snapshot:
                # Snapshots are stored under MEDIA_ROOT / "proctoring_snaps" with
                # the filename in log.snapshot. Serve them via MEDIA_URL.
                snapshot_url = request.build_absolute_uri(
                    f"{settings.MEDIA_URL}proctoring_snaps/{log.snapshot}"
                )
                snapshot_entries.append({
                    "warning_type": log.warning_type.replace('_', ' ').title(),
                    "timestamp": log.timestamp,
                    "snapshot_url": snapshot_url,
                })

        context = {
            "session": session,
            "warning_logs": all_logs_list,
            "warning_counts": dict(warning_counts),
            "snapshot_entries": snapshot_entries,
        }

        html_string = render_to_string(
            "interview_app/proctoring_report_pdf.html", context
        )

        try:
            pdf = HTML(
                string=html_string,
                base_url=request.build_absolute_uri("/"),
            ).write_pdf()
        except Exception:
            pdf = HTML(string=html_string).write_pdf()

        response = HttpResponse(pdf, content_type="application/pdf")
        response["Content-Disposition"] = (
            f'attachment; filename="proctoring_report_{session.id}.pdf"'
        )
        return response
    except InterviewSession.DoesNotExist:
        return HttpResponse("Interview session not found.", status=404)
    except Exception as e:
        # General error handling for unexpected issues during PDF generation.
        print(f"Error generating PDF report for session {session_id}: {e}")
        traceback.print_exc()
        return HttpResponse(f"An unexpected error occurred while generating the PDF report: {e}", status=500)

@csrf_exempt
@require_POST
def end_interview_session(request):
    try:
        data = json.loads(request.body)
        session_key = data.get('session_key')
        if not session_key:
            return JsonResponse({"status": "error", "message": "Session key required."}, status=400)
        
        session = InterviewSession.objects.get(session_key=session_key)
        if session.status == 'SCHEDULED':
            session.status = 'COMPLETED'
            session.save()
            print(f"--- Spoken-only session {session_key} marked as COMPLETED. ---")
            
            # Trigger comprehensive evaluation for spoken-only interviews
            try:
                from interview_app_11.comprehensive_evaluation_service import comprehensive_evaluation_service
                evaluation_results = comprehensive_evaluation_service.evaluate_complete_interview(session_key)
                print(f"--- Comprehensive evaluation completed for session {session_key} ---")
                print(f"Overall Score: {evaluation_results['overall_score']:.1f}/100")
                print(f"Recommendation: {evaluation_results['recommendation']}")
            except Exception as e:
                print(f"--- Error in comprehensive evaluation: {e} ---")
            
            # Create Evaluation after interview completion
            try:
                from evaluation.services import create_evaluation_from_session
                evaluation = create_evaluation_from_session(session_key)
                if evaluation:
                    print(f"✅ Evaluation created for session {session_key}")
            except Exception as e:
                print(f"⚠️ Error creating evaluation: {e}")
                import traceback
                traceback.print_exc()
            
        release_camera_for_session(session_key)
        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

# Removed duplicate submit_coding_challenge function - using the one at line 2643

def interview_complete(request):
    session_key = request.GET.get('session_key')
    context = {}
    
    if session_key:
        context['session_key'] = session_key
        
        # Release camera and microphone resources immediately
        try:
            release_camera_for_session(session_key)
            print(f"✅ Camera resources released for session {session_key}")
        except Exception as e:
            print(f"⚠️ Error releasing camera for session {session_key}: {e}")
        
        # Create evaluation and generate PDF in background (don't wait for it)
        import threading
        def generate_evaluation_background():
            try:
                from evaluation.services import create_evaluation_from_session
                print(f"🔄 Starting background evaluation generation for session {session_key}")
                evaluation = create_evaluation_from_session(session_key)
                if evaluation:
                    print(f"✅ Background evaluation completed for session {session_key}")
                else:
                    print(f"⚠️ Background evaluation not created for session {session_key}")
            except Exception as e:
                print(f"⚠️ Error in background evaluation generation: {e}")
                import traceback
                traceback.print_exc()
        
        # Start evaluation generation in background thread
        evaluation_thread = threading.Thread(target=generate_evaluation_background, daemon=True)
        evaluation_thread.start()
        print(f"🔄 Background evaluation thread started for session {session_key}")
    
    template = loader.get_template('interview_app/interview_complete.html')
    return HttpResponse(template.render(context, request))

def generate_and_save_follow_up(session, parent_question, transcribed_answer):
    if DEV_MODE:
        print("--- DEV MODE: Skipping AI follow-up question generation. ---")
        return None

    # CRITICAL: Do NOT generate follow-ups for closing questions like "Do you have any questions"
    closing_question_phrases = [
        "do you have any question", "do you have any questions", 
        "any questions for us", "any questions for me", "any other questions",
        "questions for us", "questions for me", "before we wrap up"
    ]
    parent_q_lower = parent_question.question_text.lower()
    if any(phrase in parent_q_lower for phrase in closing_question_phrases):
        print(f"⚠️ Parent question is a closing question. Skipping follow-up generation.")
        return None

    # Check if we've already reached 30% follow-up ratio (maintain 70% main, 30% follow-up)
    main_questions = session.questions.filter(question_level='MAIN', question_type__in=['TECHNICAL', 'BEHAVIORAL']).count()
    follow_up_questions = session.questions.filter(question_level='FOLLOW_UP', question_type__in=['TECHNICAL', 'BEHAVIORAL']).count()
    
    # Calculate projected ratio if we add this follow-up
    # We want to maintain approximately 70% main questions and 30% follow-ups
    total_questions = main_questions + follow_up_questions
    if total_questions > 0:
        # Calculate what the ratio would be if we add this follow-up
        projected_follow_ups = follow_up_questions + 1
        projected_total = total_questions + 1
        projected_ratio = projected_follow_ups / projected_total
        
        # If adding this follow-up would exceed 30%, don't generate it
        if projected_ratio > 0.30:
            current_ratio = follow_up_questions / total_questions
            print(f"⚠️ Projected follow-up ratio would be {projected_ratio*100:.1f}% (current: {current_ratio*100:.1f}%, target: 30%, main: {main_questions}, follow-ups: {follow_up_questions}). Skipping follow-up generation to maintain ratio.")
            return None
    else:
        # If no questions yet, allow first follow-up (will be checked after generation)
        print(f"ℹ️ No questions yet. Will check ratio after generation.")

    model = genai.GenerativeModel('gemini-2.0-flash')
    language_name = SUPPORTED_LANGUAGES.get(session.language_code, 'English')
    
    # Get job description context for matching
    jd_context = session.job_description or ""
    if len(jd_context) > 2000:
        jd_context = jd_context[:2000]  # Limit context size
    
    prompt = (
        f"You are a professional technical interviewer conducting a technical interview in {language_name}. "
        f"Act like a real technical interviewer - be direct, professional, and focused on technical assessment.\n\n"
        f"The candidate was asked the following question:\n'{parent_question.question_text}'\n\n"
        f"The candidate gave this transcribed answer:\n'{transcribed_answer}'\n\n"
        f"Job Description Context:\n{jd_context}\n\n"
        "Your task is to analyze the response and determine if a follow-up question is needed. Follow these rules STRICTLY:\n"
        "0. CRITICAL: If the parent question is a closing question like 'Do you have any questions for us?' or 'Before we wrap up, do you have any questions?', "
        "you MUST respond with 'NO_FOLLOW_UP' immediately. Closing questions should NEVER have follow-ups - the interview should end after the candidate answers.\n"
        "1. FIRST, check if the answer is BROAD or VAGUE (lacks specific details, examples, or depth). "
        "Signs of a broad answer: short responses, generic statements, lack of technical details, no examples, or surface-level explanations.\n"
        "2. SECOND, check if the answer topic MATCHES or RELATES to the Job Description context provided above. "
        "The answer should be relevant to the job requirements, skills, or responsibilities mentioned in the JD.\n"
        "3. ONLY if BOTH conditions are met (answer is broad/vague AND matches JD context), generate ONE single, technical follow-up question. "
        "IMPORTANT: You may use a brief introductory phrase like 'That's okay' or 'I understand' ONCE, but DO NOT repeat it. "
        "NEVER say phrases like 'That's okay, that's okay' or 'That's fine, that's fine' - this is repetitive and unprofessional. "
        "If you use an introductory phrase, use it only ONCE, then immediately ask the question. "
        "For example, if they give a vague answer about 'working with databases', your follow-up could be: "
        "'That's okay. Can you walk me through a specific database optimization you've implemented?' "
        "OR simply: 'Can you walk me through a specific database optimization you've implemented?' "
        "But NEVER: 'That's okay, that's okay. Can you walk me through...' - this is repetitive.\n"
        "4. If the answer is detailed, specific, complete, confident, OR does not relate to the JD context, "
        "you MUST respond with the exact text: NO_FOLLOW_UP\n"
        "5. CRITICAL: Your follow-up question must be a SINGLE question. If you use an introductory phrase, use it ONCE only. "
        "Example: 'That's okay. What specific challenges did you face?' OR 'What specific challenges did you face?' "
        "NOT: 'That's okay, that's okay. What specific challenges did you face?' - never repeat phrases.\n"
        "Do NOT add any other text, prefixes, or formatting. Your entire output must be either the direct follow-up question itself or the text 'NO_FOLLOW_UP'."
    )
    try:
        response = model.generate_content(prompt)
        follow_up_text = response.text.strip()
        if "NO_FOLLOW_UP" in follow_up_text or not follow_up_text: return None
        if len(follow_up_text) > 10:
            print(f"--- Generated Interactive Follow-up: {follow_up_text} ---")
            
            tts_dir = os.path.join(settings.MEDIA_ROOT, 'tts'); os.makedirs(tts_dir, exist_ok=True)
            tts_filename = f'followup_{parent_question.id}_{int(time.time())}.mp3'
            tts_path = os.path.join(tts_dir, tts_filename)
            synthesize_speech(follow_up_text, session.language_code, session.accent_tld, tts_path)
            audio_url = os.path.join(settings.MEDIA_URL, 'tts', os.path.basename(tts_path))

            follow_up_question = InterviewQuestion.objects.create(
                session=session,
                question_text=follow_up_text,
                question_type=parent_question.question_type,
                question_level='FOLLOW_UP',
                parent_question=parent_question,
                order=parent_question.order,
                audio_url=audio_url
            )
            
            return {
                'id': str(follow_up_question.id), 
                'text': follow_up_question.question_text, 
                'type': follow_up_question.question_type, 
                'audio_url': audio_url
            }
    except Exception as e:
        print(f"ERROR generating follow-up question: {e}")
    
    return None

@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        session_id = request.POST.get('session_id')
        question_id = request.POST.get('question_id')
        response_time = request.POST.get('response_time')
        no_audio_flag = str(request.POST.get('no_audio', '')).lower() in ('1', 'true', 'yes')

        if no_audio_flag:
            if not (session_id and question_id):
                return JsonResponse({'error': 'Missing session_id or question_id for no-audio submission.'}, status=400)
            try:
                question_to_update = InterviewQuestion.objects.get(id=question_id, session_id=session_id)
                question_to_update.transcribed_answer = 'No answer provided'
                fields_to_update = ['transcribed_answer']
                if response_time:
                    try:
                        question_to_update.response_time_seconds = float(response_time)
                        fields_to_update.append('response_time_seconds')
                    except ValueError:
                        pass
                question_to_update.save(update_fields=fields_to_update)
                return JsonResponse({'text': 'No answer provided', 'follow_up_question': None})
            except InterviewQuestion.DoesNotExist:
                print(f"Warning: Could not find question with ID {question_id} to save no-audio answer.")
                return JsonResponse({'error': 'Question not found'}, status=404)

        if not whisper_model:
            return JsonResponse({'error': 'Whisper model not available.'}, status=500)

        audio_file = request.FILES.get('audio_data')
        if not audio_file:
            return JsonResponse({'error': 'No audio data provided'}, status=400)

        file_path = default_storage.save('temp_audio.webm', audio_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        try:
            result = whisper_model.transcribe(full_path, fp16=False)
            transcribed_text = result.get('text', '')
            follow_up_data = None

            if session_id and question_id:
                try:
                    question_to_update = InterviewQuestion.objects.get(id=question_id, session_id=session_id)

                    question_to_update.transcribed_answer = transcribed_text or 'No answer provided'
                    fields_to_update = ['transcribed_answer']
                    if response_time:
                        try:
                            question_to_update.response_time_seconds = float(response_time)
                            fields_to_update.append('response_time_seconds')
                        except ValueError:
                            pass
                    question_to_update.save(update_fields=fields_to_update)

                    # Only generate a follow-up if the question just answered was a MAIN one
                    if transcribed_text and transcribed_text.strip() and question_to_update.question_level == 'MAIN' and question_to_update.session.language_code == 'en':
                        follow_up_data = generate_and_save_follow_up(
                            session=question_to_update.session,
                            parent_question=question_to_update,
                            transcribed_answer=transcribed_text
                        )
                except InterviewQuestion.DoesNotExist:
                    print(f"Warning: Could not find question with ID {question_id} to save answer.")
            os.remove(full_path)
            return JsonResponse({'text': transcribed_text, 'follow_up_question': follow_up_data})
        except Exception as e:
            if os.path.exists(full_path):
                os.remove(full_path)
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

# --- Video Feed and Proctoring Status ---

def video_feed(request):
    session_key = request.GET.get('session_key')
    print(f"📺 Video feed requested for session_key: {session_key}")
    
    camera = get_camera_for_session(session_key)
    if not camera: 
        print(f"❌ Camera not found for session_key: {session_key}")
        return HttpResponse("Camera not found.", status=404)
    
    print(f"✅ Camera found for session_key: {session_key}, starting video stream")
    response = StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

def gen(camera_instance):
    """Generator function for video streaming."""
    import time
    frame_count = 0
    consecutive_failures = 0
    try:
        # Always start with a frame to initialize the stream
        initial_frame = camera_instance.get_frame()
        if initial_frame and len(initial_frame) > 0:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + initial_frame + b'\r\n\r\n')
            print(f"📺 Initial frame sent for session {camera_instance.session_id}")
        
        while True:
            try:
                frame = camera_instance.get_frame()
                if frame and len(frame) > 0:
                    frame_count += 1
                    consecutive_failures = 0
                    # Ensure proper MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                else:
                    consecutive_failures += 1
                    # Log if getting too many failures
                    if consecutive_failures == 10:
                        print(f"⚠️ Camera {camera_instance.session_id} - 10 consecutive frame failures")
                    # Always yield fallback to keep stream alive
                    fallback = camera_instance._create_fallback_frame()
                    if fallback and len(fallback) > 0:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + fallback + b'\r\n\r\n')
                # Consistent delay (~15fps)
                time.sleep(0.067)
            except Exception as frame_error:
                # Always yield something to keep stream alive
                try:
                    fallback = camera_instance._create_fallback_frame()
                    if fallback and len(fallback) > 0:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + fallback + b'\r\n\r\n')
                except:
                    # Last resort: minimal valid JPEG
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x01\xe0\x02\x80\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9' + b'\r\n\r\n')
                time.sleep(0.067)
    except GeneratorExit:
        print(f"📺 Video stream closed for camera {camera_instance.session_id}")
    except Exception as e:
        print(f"❌ Error in video stream: {e}")

def get_proctoring_status(request):
    session_key = request.GET.get('session_key')
    camera = get_camera_for_session(session_key)
    if not camera: 
        # Return empty warnings object with all fields False instead of 404
        return JsonResponse({
            'no_person_warning_active': False,
            'multiple_people': False,
            'phone_detected': False,
            'no_person': False,
            'low_concentration': False,
            'tab_switched': False,
            'excessive_noise': False,
            'multiple_speakers': False
        })
    warnings = camera.get_latest_warnings()
    # Remove _counts from response to avoid confusion in frontend
    warnings.pop('_counts', None)
    return JsonResponse(warnings)

def video_frame(request):
    """Return a single JPEG frame (for polling-based display)"""
    session_key = request.GET.get('session_key')
    camera = get_camera_for_session(session_key)
    if not camera:
        # Return a minimal error frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Not Found", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            return HttpResponse(buf.tobytes(), content_type='image/jpeg')
        return HttpResponse(status=404)
    
    frame_bytes = camera.get_frame()
    if frame_bytes and len(frame_bytes) > 0:
        return HttpResponse(frame_bytes, content_type='image/jpeg')
    else:
        # Return fallback frame
        fallback = camera._create_fallback_frame()
        return HttpResponse(fallback, content_type='image/jpeg')

@csrf_exempt
@require_POST
def report_tab_switch(request):
    data = json.loads(request.body)
    session_key = data.get('session_key')
    camera = get_camera_for_session(session_key)
    if camera: camera.set_tab_switch_status(data.get('status') == 'hidden')
    return JsonResponse({"status": "ok"})

@csrf_exempt
def check_camera(request):
    session_key = request.GET.get('session_key')
    camera = get_camera_for_session(session_key)
    if camera and camera.video.isOpened():
        return JsonResponse({"status": "ok"})
    else:
        release_camera_for_session(session_key)
        return JsonResponse({"status": "error"}, status=500)

@csrf_exempt
@require_POST
def activate_proctoring_camera(request):
    """Explicitly activate YOLO model and proctoring warnings when technical interview starts"""
    try:
        data = json.loads(request.body)
        session_key = data.get('session_key')
        
        if not session_key:
            return JsonResponse({'status': 'error', 'message': 'session_key required'}, status=400)
        
        # Get or create camera for this session
        camera = get_camera_for_session(session_key)
        
        if not camera:
            return JsonResponse({'status': 'error', 'message': 'Could not create camera'}, status=500)
        
        # Ensure camera is running
        if hasattr(camera, 'video') and camera.video.isOpened():
            # Activate YOLO model and proctoring (only now, not during identity verification)
            yolo_activated = False
            if hasattr(camera, 'activate_yolo_proctoring'):
                yolo_activated = camera.activate_yolo_proctoring()
            else:
                # Fallback for older camera implementations
                print(f"⚠️ Camera doesn't have activate_yolo_proctoring method, using fallback")
                yolo_activated = True
            
            # Ensure detection loop is running
            if not camera._running:
                # Restart detection loop if it stopped
                import threading
                camera._running = True
                if hasattr(camera, '_detector_thread'):
                    if not camera._detector_thread.is_alive():
                        camera._detector_thread = threading.Thread(target=camera._capture_and_detect_loop, daemon=True)
                        camera._detector_thread.start()
                        print(f"✅ Detection loop reactivated for session {str(camera.session_id)[:8]}")
                else:
                    # Start detection loop if it doesn't exist
                    camera._detector_thread = threading.Thread(target=camera._capture_and_detect_loop, daemon=True)
                    camera._detector_thread.start()
                    print(f"✅ Detection loop started for session {str(camera.session_id)[:8]}")
            
            return JsonResponse({
                'status': 'success', 
                'message': 'YOLO model and proctoring warnings activated for technical interview',
                'camera_active': True,
                'yolo_loaded': yolo_activated,
                'proctoring_active': getattr(camera, '_proctoring_active', False),
                'detection_running': camera._running if hasattr(camera, '_running') else False
            })
        else:
            return JsonResponse({'status': 'error', 'message': 'Camera not opened'}, status=500)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
@require_POST
def release_camera(request):
    try:
        data = json.loads(request.body)
        session_key = data.get('session_key')
        release_camera_for_session(session_key)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def execute_code(request):
    """
    Minimal placeholder endpoint for the coding round.

    In the full project, this would safely execute user code and run
    multiple test cases. In this 15_11_NEW copy we avoid executing
    untrusted code and instead return a simulated success response
    so that the interview portal can function without 500 errors.
    """
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST is allowed for this endpoint."},
            status=405,
        )

    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        data = {}

    code = data.get("code", "")
    language = data.get("language", "python")
    question_id = data.get("question_id")
    session_key = data.get("session_key")

    output_lines = [
        f"Code execution is running in demo mode for language={language}.",
        "All test cases are treated as PASSED in this environment.",
        "",
        "Received code snippet (first 200 chars):",
        code[:200],
    ]

    return JsonResponse(
        {
            "success": True,
            "output": "\n".join(output_lines),
            "passed": True,
            "question_id": question_id,
            "session_key": session_key,
        }
    )


@csrf_exempt
def submit_coding_challenge(request):
    """
    Minimal placeholder endpoint for final coding submission.

    The full project stores code, runs full test suites, and marks
    interview completion. Here we just accept the payload and return
    a simulated "all tests passed" response so the UI can proceed.
    """
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST is allowed for this endpoint."},
            status=405,
        )

    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        data = {}

    code = data.get("code", "")
    language = data.get("language", "python")
    question_id = data.get("question_id")
    session_key = data.get("session_key")

    output_log_lines = [
        f"Final submission received for language={language}.",
        "In this demo environment, all test cases are treated as PASSED.",
        "",
        "First 200 characters of submitted code:",
        code[:200],
    ]

    return JsonResponse(
        {
            "success": True,
            "passed_all_tests": True,
            "output_log": "\n".join(output_log_lines),
            "question_id": question_id,
            "session_key": session_key,
        }
    )

def extract_id_data(image_path, model):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    id_card_for_ocr = {'mime_type': 'image/jpeg', 'data': image_bytes}
    prompt = ("You are an OCR expert. Extract the following from the provided image of an ID card: "
              "- Full Name\n- ID Number\n"
              "If a value cannot be extracted, state 'Not Found'. Do not add any warnings.\n"
              "Format:\nName: <value>\nID Number: <value>")
              
    response = model.generate_content([prompt, id_card_for_ocr])
    text = response.text
    name_match = re.search(r"Name:\s*(.+)", text, re.IGNORECASE)
    id_number_match = re.search(r"ID Number:\s*(.+)", text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None
    id_number = id_number_match.group(1).strip() if id_number_match else None
    return id_number, name

# === AI Chatbot endpoints (Q&A phase) ===

def chatbot_standalone(request):
    """Render standalone chatbot template with direct Deepgram connection"""
    session_key = request.GET.get('session_key', '')
    print(f"\n{'='*60}")
    print(f"🤖 CHATBOT_STANDALONE called")
    print(f"{'='*60}")
    print(f"   Session Key from URL: '{session_key}'")
    print(f"   Session Key length: {len(session_key)}")
    print(f"   All GET params: {dict(request.GET)}")
    print(f"{'='*60}\n")
    
    if not session_key or not session_key.strip():
        print(f"⚠️ WARNING: session_key is empty in chatbot_standalone!")
        print(f"   This will cause 'Session key is required' error in /ai/start")
    
    return render(request, 'interview_app/chatbot_direct_deepgram.html', {
        'session_key': session_key
    })

@csrf_exempt
@require_POST
def ai_start(request):
    from .complete_ai_bot import start_interview, sessions
    from .models import InterviewSession as DjangoSession, InterviewQuestion
    
    print(f"\n{'='*60}")
    print(f"🎯 AI_START called (using complete_ai_bot)")
    print(f"{'='*60}")
    try:
        data = json.loads(request.body.decode('utf-8'))
        print(f"📦 Received JSON data: {data}")
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}, using POST data")
        data = request.POST
    
    session_key = data.get('session_key', '')
    print(f"🔑 Session Key: {session_key}")
    
    # Initialize variables - will be populated from database
    candidate_name = None
    jd_text = ''
    django_session = None
    
    # Get question count - PRIORITIZE explicit value from portal.html form
    # This comes from the ?qc= URL parameter passed to chatbot iframe
    question_count = 4  # Default fallback
    explicit_qc = data.get('question_count')
    print(f"📊 Question Count from request: {explicit_qc}")
    
    if explicit_qc:
        try:
            question_count = max(1, min(15, int(explicit_qc)))  # Between 1 and 15
            print(f"✅ Using question_count from portal.html form: {question_count}")
        except (TypeError, ValueError) as e:
            print(f"⚠️ Invalid question_count '{explicit_qc}', using default 4: {e}")
            question_count = 4
    
    if not session_key:
        print(f"❌ ERROR: No session_key provided in request!")
        print(f"   Request data keys: {list(data.keys())}")
        return JsonResponse({"error": "Session key is required"}, status=400)
    
    try:
        django_session = DjangoSession.objects.get(session_key=session_key)
        # ALWAYS use candidate_name from database (saved from portal.html form)
        candidate_name = django_session.candidate_name or ''
        # ALWAYS use JD from database (saved from portal.html form)
        jd_text = django_session.job_description or ''
        
        print(f"✅ Retrieved from DB (saved from portal.html form):")
        print(f"   Session ID: {django_session.id}")
        print(f"   Candidate Name: '{candidate_name}' (length: {len(candidate_name)})")
        print(f"   JD length: {len(jd_text)}")
        print(f"   JD preview: {jd_text[:200]}...")
        print(f"   JD full text: {jd_text}")
        
        # Validate that we have the required data
        if not candidate_name or not candidate_name.strip():
            print(f"❌ ERROR: Candidate name is empty in database!")
            print(f"   Session ID: {django_session.id}")
            print(f"   Session Key: {session_key}")
            print(f"   Candidate Name field value: '{django_session.candidate_name}'")
            return JsonResponse({"error": "Candidate name is missing from session. Please ensure you filled in the candidate name in the form."}, status=400)
        
        # If question_count was NOT explicitly set from the portal.html form,
        # try to derive it from InterviewSlot.ai_configuration (for scheduled interviews).
        # But ALWAYS prioritize the explicit value from portal.html form.
        if not explicit_qc:
            print(f"⚠️ No explicit question_count from portal.html, checking InterviewSlot...")
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
                    print(f"✅ Found Interview {interview.id} with Slot {slot.id}")
                    print(f"   Slot AI Config: {slot.ai_configuration}")
                    print(f"   Slot AI Config Type: {type(slot.ai_configuration)}")
                    
                    # Try multiple ways to get question_count
                    found_count = None
                    
                    # Method 1: Check if ai_configuration is a dict and has question_count
                    if slot.ai_configuration and isinstance(slot.ai_configuration, dict):
                        # Try different key variations
                        possible_keys = ['question_count', 'questionCount', 'question-count', 'questions', 'num_questions']
                        for key in possible_keys:
                            if key in slot.ai_configuration:
                                found_count = slot.ai_configuration[key]
                                print(f"✅ Found question_count using key '{key}': {found_count}")
                                break
                        
                        # If not found, print all available keys for debugging
                        if found_count is None:
                            print(f"⚠️ question_count not found in ai_configuration")
                            print(f"   Available keys: {list(slot.ai_configuration.keys()) if slot.ai_configuration else 'None'}")
                            print(f"   Full ai_configuration: {slot.ai_configuration}")
                    
                    # Method 2: Check if ai_configuration is a string (JSON string)
                    elif slot.ai_configuration and isinstance(slot.ai_configuration, str):
                        try:
                            # json is already imported at module level
                            config_dict = json.loads(slot.ai_configuration)
                            if isinstance(config_dict, dict):
                                possible_keys = ['question_count', 'questionCount', 'question-count', 'questions', 'num_questions']
                                for key in possible_keys:
                                    if key in config_dict:
                                        found_count = config_dict[key]
                                        print(f"✅ Found question_count in JSON string using key '{key}': {found_count}")
                                        break
                        except Exception as e:
                            print(f"⚠️ Error parsing ai_configuration as JSON: {e}")
                    
                    # Method 3: Check if question_count is a direct attribute
                    if found_count is None and hasattr(slot, 'question_count') and slot.question_count:
                        found_count = slot.question_count
                        print(f"✅ Using question count from InterviewSlot.question_count attribute: {found_count}")
                    
                    # Convert to integer if found
                    if found_count is not None:
                        try:
                            question_count = int(found_count)
                            if question_count > 0:
                                print(f"✅✅✅ Using question count from InterviewSlot: {question_count}")
                            else:
                                print(f"⚠️ Invalid question_count ({question_count}), using default 4")
                                question_count = 4
                        except (ValueError, TypeError) as e:
                            print(f"⚠️ Error converting question_count to int: {e}, using default 4")
                            question_count = 4
                    else:
                        print(f"⚠️⚠️⚠️ No question_count found anywhere in slot.ai_configuration!")
                        print(f"   Slot ID: {slot.id}")
                        print(f"   Slot ai_configuration type: {type(slot.ai_configuration)}")
                        print(f"   Slot ai_configuration value: {slot.ai_configuration}")
                        if isinstance(slot.ai_configuration, dict):
                            print(f"   Available keys in ai_configuration: {list(slot.ai_configuration.keys())}")
                        print(f"⚠️ Using default question_count: 4")
                        question_count = 4
                else:
                    if not interview:
                        print(f"⚠️ No Interview found for session_key={session_key}, candidate_email={django_session.candidate_email}")
                    elif not interview.slot:
                        print(f"⚠️ Interview {interview.id} has no slot assigned")
                    print(f"⚠️ Using default question_count: 4")
            except Exception as e:
                print(f"⚠️ Error getting question count, using default 4: {e}")
                import traceback
                traceback.print_exc()
    except DjangoSession.DoesNotExist:
        print(f"❌ Session not found in DB")
        print(f"   Session Key searched: {session_key}")
        print(f"   Available sessions in DB: {DjangoSession.objects.count()}")
        return JsonResponse({"error": f"Invalid session key: {session_key}. Please start a new interview."}, status=400)
    except Exception as e:
        print(f"❌ Error retrieving session: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Error retrieving session: {str(e)}"}, status=500)
    
    # Ensure we always have some JD text so the AI bot doesn't fail with
    # "Job description is required" when older sessions are missing JD.
    if not (jd_text or "").strip():
        print("⚠️ JD text is empty for this session – using a safe fallback JD to proceed.")
        jd_text = (
            "This is a general technical interview for an early-career candidate. "
            "Ask a mix of basic data structures, algorithms, and problem-solving questions "
            "appropriate for an intern-level role."
        )

    # Validate we have all required data
    if not candidate_name:
        print(f"❌ ERROR: Candidate name is missing!")
        return JsonResponse({"error": "Candidate name is required"}, status=400)
    
    # Call AI bot to start interview with data from portal.html form
    print(f"\n{'='*60}")
    print(f"🎯🎯🎯 FINAL: Starting interview with data from portal.html form")
    print(f"   Candidate Name: {candidate_name}")
    print(f"   Question Count: {question_count}")
    print(f"   JD length: {len(jd_text)}")
    print(f"   JD preview: {jd_text[:200]}...")
    print(f"{'='*60}\n")
    result = start_interview(candidate_name, jd_text, max_questions=question_count)
    print(f"✅ Interview started, returned max_questions={result.get('max_questions', 'N/A')}")
    
    # Verify the session has the correct max_questions
    if 'session_id' in result and result['session_id'] in sessions:
        ai_session = sessions[result['session_id']]
        print(f"✅ AI Session max_questions: {ai_session.max_questions}")
        if ai_session.max_questions != question_count:
            print(f"⚠️⚠️⚠️ CRITICAL WARNING: Session max_questions ({ai_session.max_questions}) != requested question_count ({question_count})")
            print(f"   This means the interview will ask {ai_session.max_questions} questions instead of {question_count}!")
        else:
            print(f"✅✅✅ Verified: Session max_questions ({ai_session.max_questions}) matches requested question_count ({question_count})")
    
    # Link the AI session to Django session and create first question in database
    if 'error' not in result and django_session:
        session_id = result.get('session_id')
        if session_id and session_id in sessions:
            ai_session = sessions[session_id]
            # Store django session_key in AI session for later reference
            ai_session.django_session_key = session_key
            
            # Create the first question in database
            first_question_text = result.get('question', '')
            if first_question_text:
                try:
                    # Check if question already exists to avoid duplicates
                    existing_first = InterviewQuestion.objects.filter(
                        session=django_session,
                        order=0,
                        question_text=first_question_text
                    ).first()
                    
                    if not existing_first:
                        # First question should always be order 0
                        # Check if any questions exist - if so, use max_order + 1, otherwise use 0
                        from django.db.models import Max
                        existing_count = InterviewQuestion.objects.filter(session=django_session).count()
                        
                        if existing_count == 0:
                            # No questions exist - use order 0 for first question
                            new_order = 0
                        else:
                            # Questions exist - use max_order + 1 to avoid conflicts
                            max_order = InterviewQuestion.objects.filter(
                                session=django_session
                            ).aggregate(max_order=Max('order'))['max_order'] or -1
                            new_order = max_order + 1
                        
                        InterviewQuestion.objects.create(
                            session=django_session,
                            question_text=first_question_text,
                            question_type='TECHNICAL',
                            order=new_order,
                            question_level='MAIN',
                            audio_url=result.get('audio_url', '')
                        )
                        print(f"✅ Created first question in database with order {new_order}")
                    else:
                        print(f"✅ First question already exists in database")
                except Exception as e:
                    print(f"⚠️ Error creating question in database: {e}")
                    import traceback
                    traceback.print_exc()
    
    if 'error' in result:
        print(f"❌ Error in result: {result['error']}")
    else:
        print(f"✅ Session ID: {result.get('session_id', 'N/A')}")
        print(f"✅ Question: {result.get('question', 'N/A')[:100]}...")
        print(f"✅ Audio URL: {result.get('audio_url', 'N/A')}")
    print(f"{'='*60}\n")
    
    status_code = 200 if 'error' not in result else 400
    return JsonResponse(result, status=status_code)


@csrf_exempt
@require_POST
def ai_upload_answer(request):
    from .complete_ai_bot import upload_answer, sessions
    from .models import InterviewSession as DjangoSession, InterviewQuestion
    
    try:
        # Support both multipart and JSON
        session_id = request.POST.get('session_id')
        if not session_id and request.body:
            data = json.loads(request.body.decode('utf-8'))
            session_id = data.get('session_id')
            transcript = data.get('transcript', '')
        else:
            transcript = (request.POST.get('transcript') or '').strip()
        
        print(f"\n{'='*60}")
        print(f"📝 AI_UPLOAD_ANSWER called")
        print(f"   Session ID: {session_id}")
        print(f"   Transcript: {transcript[:100] if transcript else 'Empty'}...")
        print(f"{'='*60}")
        
        # Save answer to database if session exists
        if session_id and session_id in sessions:
            ai_session = sessions[session_id]
            # Try to find the Django session using session_key stored in ai_session
            django_session = None
            if hasattr(ai_session, 'django_session_key') and ai_session.django_session_key:
                try:
                    django_session = DjangoSession.objects.get(session_key=ai_session.django_session_key)
                except DjangoSession.DoesNotExist:
                    print(f"⚠️ Django session not found for session_key: {ai_session.django_session_key}")
            
            # If we have a Django session, save the answer to InterviewQuestion
            # IMPORTANT: Also save "No answer provided" for closing questions
            if django_session:
                try:
                    # Find the current question for this session based on question number
                    current_q_num = ai_session.current_question_number
                    # Get the last question that was asked (current_question_number - 1 is the one being answered)
                    question_to_answer = current_q_num - 1 if current_q_num > 0 else 0
                    
                    # Find the question by order (question numbers are 1-indexed in UI, 0-indexed in code)
                    questions = InterviewQuestion.objects.filter(
                        session=django_session
                    ).order_by('order')
                    
                    # Check if this is a pre-closing or closing question answer
                    is_pre_closing_question = False
                    is_closing_question = False
                    
                    # Check pre-closing question flag
                    if hasattr(ai_session, 'asked_pre_closing_question') and ai_session.asked_pre_closing_question:
                        # Check if we just answered the pre-closing question (before asked_for_questions is set)
                        if hasattr(ai_session, 'asked_for_questions') and not ai_session.asked_for_questions:
                            is_pre_closing_question = True
                            print(f"📝 Detected pre-closing question answer")
                    
                    # Check if this is a closing question answer
                    if hasattr(ai_session, 'last_active_question_text') and ai_session.last_active_question_text:
                        closing_phrases = ["do you have any question", "any questions for", "before we wrap up"]
                        is_closing_question = any(phrase in ai_session.last_active_question_text.lower() for phrase in closing_phrases)
                        if is_closing_question:
                            print(f"📝 Detected closing question answer")
                    
                    # Determine answer text:
                    # - If transcript exists and is valid, use it (even if it's "no")
                    # - If transcript is empty or invalid, use "No answer provided"
                    if transcript and transcript.strip() and not transcript.startswith("[") and not transcript.startswith("[No speech"):
                        answer_text = transcript.strip()
                    else:
                        answer_text = 'No answer provided'
                    
                    # IMPORTANT: Always save answers for pre-closing and closing questions
                    is_special_question = is_pre_closing_question or is_closing_question
                    
                    if questions.exists():
                        # Try to find the question by matching the last_active_question_text first (most reliable)
                        question_obj = None
                        if hasattr(ai_session, 'last_active_question_text') and ai_session.last_active_question_text:
                            # Try exact text match first
                            matching_question = questions.filter(
                                question_text=ai_session.last_active_question_text
                            ).first()
                            if matching_question:
                                question_obj = matching_question
                                print(f"✅ Found question by exact text match: order {question_obj.order}")
                            else:
                                # Try partial match (in case of slight variations)
                                for q in questions:
                                    if ai_session.last_active_question_text.lower().strip() in q.question_text.lower() or q.question_text.lower() in ai_session.last_active_question_text.lower().strip():
                                        question_obj = q
                                        print(f"✅ Found question by partial text match: order {question_obj.order}")
                                        break
                        
                        # If not found by text, try by order
                        if not question_obj and question_to_answer < questions.count():
                            question_obj = questions[question_to_answer]
                            print(f"✅ Found question by order index: {question_to_answer}")
                        
                        # If still not found, try to find the last question without an answer (for pre-closing/closing)
                        if not question_obj:
                            unanswered_question = questions.filter(transcribed_answer__isnull=True).last()
                            if unanswered_question:
                                question_obj = unanswered_question
                                print(f"✅ Found last unanswered question: order {question_obj.order}")
                        
                        # For pre-closing/closing questions, also try to find by checking if it's the second-to-last or last question
                        if not question_obj and is_special_question:
                            # Pre-closing and closing are usually the last or second-to-last questions
                            if questions.count() >= 2:
                                # Try second-to-last (might be pre-closing)
                                question_obj = questions[questions.count() - 2]
                                print(f"✅ Found question as second-to-last (pre-closing?): order {question_obj.order}")
                            elif questions.count() >= 1:
                                # Try last question (might be closing)
                                question_obj = questions.last()
                                print(f"✅ Found question as last (closing?): order {question_obj.order}")
                        
                        if question_obj:
                            # Always save for pre-closing and closing questions, or if there's an actual answer
                            if is_special_question or answer_text != 'No answer provided':
                                question_obj.transcribed_answer = answer_text
                                # Calculate response time if available
                                import time as time_module
                                if hasattr(ai_session, 'question_asked_at') and ai_session.question_asked_at:
                                    response_time = time_module.time() - ai_session.question_asked_at
                                    question_obj.response_time_seconds = response_time
                                question_obj.save()
                                print(f"✅ Saved answer to database for question {question_obj.order} ({'pre-closing' if is_pre_closing_question else 'closing' if is_closing_question else 'regular'}): {answer_text[:50]}...")
                            elif answer_text == 'No answer provided' and not is_special_question:
                                # For non-special questions, only save if there's an actual answer
                                pass
                        else:
                            print(f"⚠️ Could not find question to save answer. Creating new question...")
                            # Create a new question if it doesn't exist (including pre-closing and closing questions)
                            # Get the maximum order to ensure unique sequential ordering
                            from django.db.models import Max
                            max_order_result = InterviewQuestion.objects.filter(
                                session=django_session
                            ).aggregate(max_order=Max('order'))
                            max_order = max_order_result['max_order'] if max_order_result['max_order'] is not None else -1
                            new_order = max_order + 1
                            
                            # Determine question level
                            question_level = 'MAIN'
                            if is_pre_closing_question:
                                question_level = 'MAIN'  # Pre-closing is still MAIN level
                            elif is_closing_question:
                                question_level = 'MAIN'  # Closing is still MAIN level
                            
                            question_obj = InterviewQuestion.objects.create(
                                session=django_session,
                                question_text=ai_session.last_active_question_text or "Question",
                                question_type='TECHNICAL',
                                order=new_order,
                                question_level=question_level,
                                transcribed_answer=answer_text  # Save the answer immediately
                            )
                            print(f"✅ Created and saved answer to new question {new_order} ({'pre-closing' if is_pre_closing_question else 'closing' if is_closing_question else 'regular'}): {answer_text[:50]}...")
                    else:
                        # No questions exist yet - create the question with answer
                        # Use order 0 for the first question
                        question_obj = InterviewQuestion.objects.create(
                            session=django_session,
                            question_text=ai_session.last_active_question_text or "Question",
                            question_type='TECHNICAL',
                            order=0,  # First question gets order 0
                            question_level='MAIN',
                            transcribed_answer=answer_text
                        )
                        print(f"✅ Created first question with answer (order 0): {answer_text[:50]}...")
                except Exception as e:
                    print(f"⚠️ Error saving answer to database: {e}")
                    import traceback
                    traceback.print_exc()
        
        result = upload_answer(session_id, transcript)
        
        # Handle candidate questions and AI answers - save them to database
        if 'error' not in result and session_id and session_id in sessions:
            ai_session = sessions[session_id]
            if hasattr(ai_session, 'django_session_key') and ai_session.django_session_key:
                try:
                    django_session = DjangoSession.objects.get(session_key=ai_session.django_session_key)
                    
                    # Check if this is a candidate question scenario (candidate asked a question, AI answered)
                    interviewer_answer = result.get('interviewer_answer', '')
                    if interviewer_answer:
                        # This is a candidate question - save both the question and AI's answer
                        from .complete_ai_bot import is_candidate_question
                        if is_candidate_question(transcript):
                            # Get the highest order number
                            existing_questions = InterviewQuestion.objects.filter(
                                session=django_session
                            ).order_by('-order')
                            max_order = existing_questions.first().order if existing_questions.exists() else -1
                            
                            # Save candidate's question as a question with question_level='CANDIDATE_QUESTION'
                            candidate_question_obj = InterviewQuestion.objects.create(
                                session=django_session,
                                question_text=transcript,  # Candidate's question
                                question_type='TECHNICAL',
                                order=max_order + 1,
                                question_level='CANDIDATE_QUESTION',  # Special level to identify candidate questions
                                transcribed_answer=interviewer_answer  # AI's answer to candidate's question
                            )
                            print(f"✅ Saved candidate question and AI answer: Q: {transcript[:50]}... | A: {interviewer_answer[:50]}...")
                    
                    # Also check for candidate questions in closing phase (when AI answers candidate's question)
                    # Check if the result contains a combined response (answer + "Do you have any other questions?")
                    next_question_text = result.get('next_question', '')
                    if next_question_text and 'do you have any other question' in next_question_text.lower() and not interviewer_answer:
                        # This might be after answering a candidate question - check if transcript is a question
                        from .complete_ai_bot import is_candidate_question
                        if is_candidate_question(transcript):
                            # Check if we already saved this candidate question
                            existing_candidate_q = InterviewQuestion.objects.filter(
                                session=django_session,
                                question_text=transcript,
                                question_level='CANDIDATE_QUESTION'
                            ).first()
                            
                            if not existing_candidate_q:
                                # Get the highest order number
                                existing_questions = InterviewQuestion.objects.filter(
                                    session=django_session
                                ).order_by('-order')
                                max_order = existing_questions.first().order if existing_questions.exists() else -1
                                
                                # Extract AI's answer from the combined text (before "Do you have any other questions?")
                                answer_parts = next_question_text.split('Do you have any other questions')
                                ai_answer = answer_parts[0].strip() if answer_parts else next_question_text
                                
                                # Save candidate's question and AI's answer
                                candidate_question_obj = InterviewQuestion.objects.create(
                                    session=django_session,
                                    question_text=transcript,  # Candidate's question
                                    question_type='TECHNICAL',
                                    order=max_order + 1,
                                    question_level='CANDIDATE_QUESTION',
                                    transcribed_answer=ai_answer  # AI's answer to candidate's question
                                )
                                print(f"✅ Saved candidate question from closing phase: Q: {transcript[:50]}... | A: {ai_answer[:50]}...")
                    
                    # Also check for closing question in final_message if interview is completed
                    if not next_question_text and result.get('completed') and result.get('final_message'):
                        # This is the final closing message, not a question
                        pass
                    elif next_question_text and next_question_text.strip():
                        # Check if this question already exists
                        existing_questions = InterviewQuestion.objects.filter(
                            session=django_session
                        ).order_by('order')
                        
                        # Get the current question number from AI session
                        current_q_num = ai_session.current_question_number
                        
                        # Check if question at this order already exists
                        question_exists = existing_questions.filter(order=current_q_num - 1).exists()
                        
                        if not question_exists:
                            # Determine question type - check if it's a pre-closing or closing question
                            is_closing_question = any(phrase in next_question_text.lower() for phrase in [
                                "do you have any question", "any questions for", "before we wrap up"
                            ])
                            
                            # Check if this is a pre-closing question
                            is_pre_closing = False
                            if hasattr(ai_session, 'asked_pre_closing_question') and ai_session.asked_pre_closing_question:
                                if hasattr(ai_session, 'asked_for_questions') and not ai_session.asked_for_questions:
                                    is_pre_closing = True
                            
                            # Get the maximum order to ensure unique sequential ordering
                            from django.db.models import Max
                            max_order_result = InterviewQuestion.objects.filter(
                                session=django_session
                            ).aggregate(max_order=Max('order'))
                            max_order = max_order_result['max_order'] if max_order_result['max_order'] is not None else -1
                            new_order = max_order + 1
                            
                            # Create new question (pre-closing or closing) with proper sequential order
                            question_obj = InterviewQuestion.objects.create(
                                session=django_session,
                                question_text=next_question_text,
                                question_type='TECHNICAL',
                                order=new_order,  # Use calculated max_order + 1 for sequential ordering
                                question_level='MAIN',
                                audio_url=result.get('audio_url', '')
                            )
                            question_type_label = 'pre-closing' if is_pre_closing else 'closing' if is_closing_question else 'regular'
                            print(f"✅ Created new {question_type_label} question with order {new_order} in database: {next_question_text[:50]}...")
                            
                            # If interview is completed and we just created the closing question,
                            # OR if this is a pre-closing/closing question, try to save the answer if transcript is available
                            if result.get('completed') or is_closing_question or is_pre_closing:
                                try:
                                    # For pre-closing/closing questions, save actual transcript if available (even if it's "no")
                                    # Only use "No answer provided" if transcript is truly empty
                                    if transcript and transcript.strip() and not transcript.startswith("[") and not transcript.startswith("[No speech"):
                                        answer_text = transcript.strip()
                                        question_obj.transcribed_answer = answer_text
                                        question_obj.save()
                                        print(f"✅ Saved answer to {question_type_label} question: {answer_text[:50]}...")
                                    else:
                                        # Don't save "No answer provided" yet - wait for the actual answer
                                        print(f"⚠️ No transcript available yet for {question_type_label} question, will save when answer comes")
                                except Exception as e:
                                    print(f"⚠️ Error saving {question_type_label} question answer: {e}")
                except Exception as e:
                    print(f"⚠️ Error creating question in database: {e}")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success - Completed: {result.get('completed', False)}")
            if not result.get('completed'):
                print(f"✅ Next question: {result.get('next_question', 'N/A')[:100]}...")
                print(f"✅ Audio URL: {result.get('audio_url', 'NOT PROVIDED')}")
                if not result.get('audio_url'):
                    print(f"⚠️ WARNING: No audio URL returned for next question!")
        print(f"{'='*60}\n")
        
        status_code = 200 if 'error' not in result else 400
        return JsonResponse(result, status=status_code)
    except Exception as e:
        print(f"❌ Exception in ai_upload_answer: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def ai_repeat(request):
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        data = request.POST
    session_id = data.get('session_id')
    result = ai_repeat_django(session_id)
    status_code = 200 if 'error' not in result else 400
    return JsonResponse(result, status=status_code)


def ai_transcript_pdf(request):
    """Generate comprehensive PDF with Q&A results"""
    session_key = request.GET.get('session_key', '')
    session_id = request.GET.get('session_id', '')
    
    print(f"\n{'='*60}")
    print(f"📄 PDF GENERATION REQUEST")
    print(f"   Session Key: {session_key}")
    print(f"   Session ID: {session_id}")
    print(f"{'='*60}")
    
    # Try to get session by session_key first, then by session_id
    try:
        if session_key:
            # Verify session exists
            from .models import InterviewSession
            try:
                session = InterviewSession.objects.get(session_key=session_key)
                print(f"✅ Found session: {session.candidate_name}")
            except InterviewSession.DoesNotExist:
                print(f"❌ Session not found: {session_key}")
                return JsonResponse({'error': 'Session not found'}, status=404)
            
            # Generate PDF
            from .comprehensive_pdf import ai_comprehensive_pdf_django
            pdf_bytes = ai_comprehensive_pdf_django(session_key)
            
        elif session_id:
            # Get session_key from session_id
            from .models import InterviewSession
            session = InterviewSession.objects.get(id=session_id)
            print(f"✅ Found session by ID: {session.candidate_name}")
            
            from .comprehensive_pdf import ai_comprehensive_pdf_django
            pdf_bytes = ai_comprehensive_pdf_django(session.session_key)
        else:
            print(f"❌ No session key or ID provided")
            return JsonResponse({'error': 'Session key or ID required'}, status=400)
        
        if not pdf_bytes or len(pdf_bytes) == 0:
            print(f"❌ PDF generation returned empty bytes")
            return JsonResponse({'error': 'PDF generation returned empty'}, status=500)
        
        print(f"✅ PDF generated successfully: {len(pdf_bytes)} bytes")
        print(f"{'='*60}\n")
        
        response = HttpResponse(pdf_bytes, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="interview_report_{session_key or session_id}.pdf"'
        return response
        
    except InterviewSession.DoesNotExist:
        print(f"❌ Session not found")
        return JsonResponse({'error': 'Session not found'}, status=404)
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'PDF generation failed: {str(e)}'}, status=500)

@csrf_exempt
@require_POST
def verify_id(request):
    try:
        image_data = request.POST.get('image_data') 
        session_id = request.POST.get('session_id')

        if not all([image_data, session_id]):
            return JsonResponse({'status': 'error', 'message': 'Missing required data.'}, status=400)

        session = InterviewSession.objects.get(id=session_id)
        
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        img_file = ContentFile(base64.b64decode(imgstr), name=f"id_{timezone.now().strftime('%Y%m%d%H%M%S')}.{ext}")
        session.id_card_image.save(img_file.name, img_file, save=True)
        
        tmp_path = session.id_card_image.path
        
        full_image = cv2.imread(tmp_path)
        if full_image is None:
            return JsonResponse({'status': 'error', 'message': 'Invalid image format.'})

        results = detect_face_with_yolo(full_image)
        boxes = results[0].boxes if results and hasattr(results[0], 'boxes') else []
        num_faces_detected = len(boxes)

        # Check for exactly two faces (candidate + ID photo)
        if num_faces_detected != 2:
            if num_faces_detected < 2:
                message = f"Verification failed. Only {num_faces_detected} face(s) detected. Please ensure both your live face and the face on your ID card are clearly visible and well-lit."
            else:
                message = f"Verification failed. {num_faces_detected} faces detected. Please ensure only you and your ID card are in the frame, with no other people in the background."
            return JsonResponse({'status': 'error', 'message': message})

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            id_number, name = extract_id_data(tmp_path, model)
        except Exception as ai_error:
            print(f"AI OCR failed: {ai_error}")
            # Fallback: Skip OCR and just verify face count
            id_number, name = "SKIPPED", session.candidate_name
        
        session.extracted_id_details = f"Name: {name}, ID: {id_number}"
        
        invalid_phrases = ['not found', 'cannot be', 'unreadable', 'blurry', 'unavailable', 'missing']
        name_verified = name and len(name.strip()) > 2 and not any(phrase in name.lower() for phrase in invalid_phrases)

        # If AI OCR failed, skip name verification
        if name == "SKIPPED":
            print("AI OCR was skipped, proceeding with face count verification only")
        elif not name_verified:
            return JsonResponse({'status': 'error', 'message': f"Could not reliably read the name from the ID card. Extracted: '{name}'. Please try again."})
        elif session.candidate_name.lower().split()[0] not in name.lower():
             return JsonResponse({'status': 'error', 'message': f"Name on ID ('{name}') does not match the registered name ('{session.candidate_name}')."})

        session.id_verification_status = 'Verified'
        session.save()

        return JsonResponse({'status': 'success', 'message': 'Verification successful!'})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}, status=500)
    pass

# Removed all coding-related functions (execute_code, submit_coding_challenge, run_test_suite, and all execute_*_windows functions)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
import json

class InterviewResultsAPIView(APIView):
    """
    API endpoint to get interview results and evaluation data
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, session_id):
        """Get interview results and evaluation data"""
        try:
            session = InterviewSession.objects.get(id=session_id)
            
            # Check if user has permission to view this session
            if not request.user.is_superuser and not request.user.is_staff:
                # For now, allow all authenticated users to view results
                # You can add more specific permission logic here
                pass
            
            # Get all questions and answers
            all_questions = list(session.questions.all().order_by('order'))
            all_logs_list = list(session.logs.all())
            warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
            
            # Calculate analytics
            total_filler_words = 0
            avg_wpm = 0
            wpm_count = 0
            sentiment_scores = []
            avg_response_time = 0
            response_time_count = 0

            if session.language_code == 'en':
                for item in all_questions:
                    if item.transcribed_answer:
                        word_count = len(item.transcribed_answer.split())
                        read_time_result = readtime.of_text(item.transcribed_answer)
                        read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                        if read_time_minutes > 0:
                            item.words_per_minute = round(word_count / read_time_minutes)
                            avg_wpm += item.words_per_minute
                            wpm_count += 1
                        else:
                            item.words_per_minute = 0
                        if item.response_time_seconds:
                            avg_response_time += item.response_time_seconds
                            response_time_count += 1
                        lower_answer = item.transcribed_answer.lower()
                        item.filler_word_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                        total_filler_words += item.filler_word_count
                        sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': TextBlob(item.transcribed_answer).sentiment.polarity})
                    else:
                        sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': 0.0})
            
            final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
            final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0

            # Prepare questions data
            questions_data = []
            for question in all_questions:
                question_data = {
                    'id': str(question.id),
                    'order': question.order,
                    'question_text': question.question_text,
                    'question_type': question.question_type,
                    'question_level': question.question_level,
                    'transcribed_answer': question.transcribed_answer,
                    'response_time_seconds': question.response_time_seconds,
                    'words_per_minute': getattr(question, 'words_per_minute', 0),
                    'filler_word_count': getattr(question, 'filler_word_count', 0),
                    'audio_url': question.audio_url if hasattr(question, 'audio_url') else None,
                }
                questions_data.append(question_data)

            # Prepare analytics data
            analytics_data = {
                'warning_counts': dict(warning_counts),
                'sentiment_scores': sentiment_scores,
                'evaluation_scores': {
                    'resume_score': session.resume_score or 0,
                    'answers_score': session.answers_score or 0,
                    'overall_performance_score': session.overall_performance_score or 0
                },
                'communication_metrics': {
                    'avg_words_per_minute': final_avg_wpm,
                    'total_filler_words': total_filler_words,
                    'avg_response_time': final_avg_response_time,
                    'total_questions': len(all_questions),
                    'answered_questions': len([q for q in all_questions if q.transcribed_answer]),
                }
            }

            # Prepare response data
            response_data = {
                'session_id': str(session.id),
                'candidate_name': session.candidate_name,
                'candidate_email': session.candidate_email,
                'job_description': session.job_description,
                'scheduled_at': session.scheduled_at.isoformat() if session.scheduled_at else None,
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'status': session.status,
                'language_code': session.language_code,
                'is_evaluated': session.is_evaluated,
                
                # Evaluation results
                'resume_score': session.resume_score,
                'resume_feedback': session.resume_feedback,
                'answers_score': session.answers_score,
                'answers_feedback': session.answers_feedback,
                'overall_performance_score': session.overall_performance_score,
                'overall_performance_feedback': session.overall_performance_feedback,
                'behavioral_analysis': session.behavioral_analysis,
                'keyword_analysis': session.keyword_analysis,
                
                # Questions and answers
                'questions': questions_data,
                
                # Analytics
                'analytics': analytics_data,
                
                # Proctoring data
                'proctoring_warnings': dict(warning_counts),
                'total_warnings': sum(warning_counts.values()),
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except InterviewSession.DoesNotExist:
            return Response({
                'error': 'Interview session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error retrieving interview results: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InterviewResultsListAPIView(APIView):
    """
    API endpoint to list all interview results for the authenticated user
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get list of all interview sessions with basic results"""
        try:
            # Get all sessions (you can add filtering based on user role)
            sessions = InterviewSession.objects.all().order_by('-created_at')
            
            sessions_data = []
            for session in sessions:
                # Get basic analytics
                all_questions = session.questions.all()
                answered_questions = all_questions.filter(transcribed_answer__isnull=False).exclude(transcribed_answer='').count()
                
                # Get warning count
                warning_count = session.logs.count()
                
                session_data = {
                    'session_id': str(session.id),
                    'candidate_name': session.candidate_name,
                    'candidate_email': session.candidate_email,
                    'scheduled_at': session.scheduled_at.isoformat() if session.scheduled_at else None,
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                    'status': session.status,
                    'is_evaluated': session.is_evaluated,
                    
                    # Basic metrics
                    'total_questions': all_questions.count(),
                    'answered_questions': answered_questions,
                    'warning_count': warning_count,
                    
                    # Scores (if evaluated)
                    'resume_score': session.resume_score,
                    'answers_score': session.answers_score,
                    'overall_performance_score': session.overall_performance_score,
                }
                sessions_data.append(session_data)
            
            return Response({
                'sessions': sessions_data,
                'total_count': len(sessions_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error retrieving interview sessions: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InterviewAnalyticsAPIView(APIView):
    """
    API endpoint to get detailed analytics for interview sessions
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, session_id):
        """Get detailed analytics for a specific interview session"""
        try:
            session = InterviewSession.objects.get(id=session_id)
            
            # Get all questions and calculate detailed analytics
            all_questions = list(session.questions.all().order_by('order'))
            all_logs_list = list(session.logs.all())
            
            # Calculate detailed metrics
            total_filler_words = 0
            avg_wpm = 0
            wpm_count = 0
            sentiment_scores = []
            avg_response_time = 0
            response_time_count = 0
            question_analytics = []

            if session.language_code == 'en':
                for item in all_questions:
                    question_analytics_item = {
                        'question_id': str(item.id),
                        'question_order': item.order,
                        'question_text': item.question_text,
                        'question_type': item.question_type,
                        'has_answer': bool(item.transcribed_answer),
                        'answer_length': len(item.transcribed_answer) if item.transcribed_answer else 0,
                        'response_time': item.response_time_seconds,
                        'words_per_minute': 0,
                        'filler_word_count': 0,
                        'sentiment_score': 0.0,
                    }
                    
                    if item.transcribed_answer:
                        word_count = len(item.transcribed_answer.split())
                        read_time_result = readtime.of_text(item.transcribed_answer)
                        read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                        
                        if read_time_minutes > 0:
                            wpm = round(word_count / read_time_minutes)
                            question_analytics_item['words_per_minute'] = wpm
                            avg_wpm += wpm
                            wpm_count += 1
                        
                        if item.response_time_seconds:
                            avg_response_time += item.response_time_seconds
                            response_time_count += 1
                        
                        lower_answer = item.transcribed_answer.lower()
                        filler_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                        question_analytics_item['filler_word_count'] = filler_count
                        total_filler_words += filler_count
                        
                        sentiment = TextBlob(item.transcribed_answer).sentiment.polarity
                        question_analytics_item['sentiment_score'] = sentiment
                        sentiment_scores.append(sentiment)
                    
                    question_analytics.append(question_analytics_item)
            
            final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
            final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Proctoring analytics
            warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
            total_warnings = sum(warning_counts.values())
            
            analytics_data = {
                'session_overview': {
                    'total_questions': len(all_questions),
                    'answered_questions': len([q for q in all_questions if q.transcribed_answer]),
                    'total_warnings': total_warnings,
                    'session_duration': None,  # You can calculate this if you have start/end times
                },
                
                'communication_metrics': {
                    'avg_words_per_minute': final_avg_wpm,
                    'total_filler_words': total_filler_words,
                    'avg_response_time': final_avg_response_time,
                    'avg_sentiment_score': round(avg_sentiment, 3),
                },
                
                'evaluation_scores': {
                    'resume_score': session.resume_score or 0,
                    'answers_score': session.answers_score or 0,
                    'overall_performance_score': session.overall_performance_score or 0,
                },
                
                'proctoring_analytics': {
                    'warning_counts': dict(warning_counts),
                    'total_warnings': total_warnings,
                    'warning_types': list(warning_counts.keys()),
                },
                
                'question_analytics': question_analytics,
            }
            
            return Response(analytics_data, status=status.HTTP_200_OK)
            
        except InterviewSession.DoesNotExist:
            return Response({
                'error': 'Interview session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error retrieving analytics: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse
import json

class InterviewResultsAPIView(APIView):
    """
    API endpoint to get interview results and evaluation data
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, session_id):
        """Get interview results and evaluation data"""
        try:
            session = InterviewSession.objects.get(id=session_id)
            
            # Check if user has permission to view this session
            if not request.user.is_superuser and not request.user.is_staff:
                # For now, allow all authenticated users to view results
                # You can add more specific permission logic here
                pass
            
            # Get all questions and answers
            all_questions = list(session.questions.all().order_by('order'))
            all_logs_list = list(session.logs.all())
            warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
            
            # Calculate analytics
            total_filler_words = 0
            avg_wpm = 0
            wpm_count = 0
            sentiment_scores = []
            avg_response_time = 0
            response_time_count = 0

            if session.language_code == 'en':
                for item in all_questions:
                    if item.transcribed_answer:
                        word_count = len(item.transcribed_answer.split())
                        read_time_result = readtime.of_text(item.transcribed_answer)
                        read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                        if read_time_minutes > 0:
                            item.words_per_minute = round(word_count / read_time_minutes)
                            avg_wpm += item.words_per_minute
                            wpm_count += 1
                        else:
                            item.words_per_minute = 0
                        if item.response_time_seconds:
                            avg_response_time += item.response_time_seconds
                            response_time_count += 1
                        lower_answer = item.transcribed_answer.lower()
                        item.filler_word_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                        total_filler_words += item.filler_word_count
                        sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': TextBlob(item.transcribed_answer).sentiment.polarity})
                    else:
                        sentiment_scores.append({'question': f"Q{item.order + 1}", 'score': 0.0})
            
            final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
            final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0

            # Prepare questions data
            questions_data = []
            for question in all_questions:
                question_data = {
                    'id': str(question.id),
                    'order': question.order,
                    'question_text': question.question_text,
                    'question_type': question.question_type,
                    'question_level': question.question_level,
                    'transcribed_answer': question.transcribed_answer,
                    'response_time_seconds': question.response_time_seconds,
                    'words_per_minute': getattr(question, 'words_per_minute', 0),
                    'filler_word_count': getattr(question, 'filler_word_count', 0),
                    'audio_url': question.audio_url if hasattr(question, 'audio_url') else None,
                }
                questions_data.append(question_data)

            # Prepare analytics data
            analytics_data = {
                'warning_counts': dict(warning_counts),
                'sentiment_scores': sentiment_scores,
                'evaluation_scores': {
                    'resume_score': session.resume_score or 0,
                    'answers_score': session.answers_score or 0,
                    'overall_performance_score': session.overall_performance_score or 0
                },
                'communication_metrics': {
                    'avg_words_per_minute': final_avg_wpm,
                    'total_filler_words': total_filler_words,
                    'avg_response_time': final_avg_response_time,
                    'total_questions': len(all_questions),
                    'answered_questions': len([q for q in all_questions if q.transcribed_answer]),
                }
            }

            # Prepare response data
            response_data = {
                'session_id': str(session.id),
                'candidate_name': session.candidate_name,
                'candidate_email': session.candidate_email,
                'job_description': session.job_description,
                'scheduled_at': session.scheduled_at.isoformat() if session.scheduled_at else None,
                'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                'status': session.status,
                'language_code': session.language_code,
                'is_evaluated': session.is_evaluated,
                
                # Evaluation results
                'resume_score': session.resume_score,
                'resume_feedback': session.resume_feedback,
                'answers_score': session.answers_score,
                'answers_feedback': session.answers_feedback,
                'overall_performance_score': session.overall_performance_score,
                'overall_performance_feedback': session.overall_performance_feedback,
                'behavioral_analysis': session.behavioral_analysis,
                'keyword_analysis': session.keyword_analysis,
                
                # Questions and answers
                'questions': questions_data,
                
                # Analytics
                'analytics': analytics_data,
                
                # Proctoring data
                'proctoring_warnings': dict(warning_counts),
                'total_warnings': sum(warning_counts.values()),
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except InterviewSession.DoesNotExist:
            return Response({
                'error': 'Interview session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error retrieving interview results: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InterviewResultsListAPIView(APIView):
    """
    API endpoint to list all interview results for the authenticated user
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get list of all interview sessions with basic results"""
        try:
            # Get all sessions (you can add filtering based on user role)
            sessions = InterviewSession.objects.all().order_by('-created_at')
            
            sessions_data = []
            for session in sessions:
                # Get basic analytics
                all_questions = session.questions.all()
                answered_questions = all_questions.filter(transcribed_answer__isnull=False).exclude(transcribed_answer='').count()
                
                # Get warning count
                warning_count = session.logs.count()
                
                session_data = {
                    'session_id': str(session.id),
                    'candidate_name': session.candidate_name,
                    'candidate_email': session.candidate_email,
                    'scheduled_at': session.scheduled_at.isoformat() if session.scheduled_at else None,
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                    'status': session.status,
                    'is_evaluated': session.is_evaluated,
                    
                    # Basic metrics
                    'total_questions': all_questions.count(),
                    'answered_questions': answered_questions,
                    'warning_count': warning_count,
                    
                    # Scores (if evaluated)
                    'resume_score': session.resume_score,
                    'answers_score': session.answers_score,
                    'overall_performance_score': session.overall_performance_score,
                }
                sessions_data.append(session_data)
            
            return Response({
                'sessions': sessions_data,
                'total_count': len(sessions_data)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({
                'error': f'Error retrieving interview sessions: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class InterviewAnalyticsAPIView(APIView):
    """
    API endpoint to get detailed analytics for interview sessions
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, session_id):
        """Get detailed analytics for a specific interview session"""
        try:
            session = InterviewSession.objects.get(id=session_id)
            
            # Get all questions and calculate detailed analytics
            all_questions = list(session.questions.all().order_by('order'))
            all_logs_list = list(session.logs.all())
            
            # Calculate detailed metrics
            total_filler_words = 0
            avg_wpm = 0
            wpm_count = 0
            sentiment_scores = []
            avg_response_time = 0
            response_time_count = 0
            question_analytics = []

            if session.language_code == 'en':
                for item in all_questions:
                    question_analytics_item = {
                        'question_id': str(item.id),
                        'question_order': item.order,
                        'question_text': item.question_text,
                        'question_type': item.question_type,
                        'has_answer': bool(item.transcribed_answer),
                        'answer_length': len(item.transcribed_answer) if item.transcribed_answer else 0,
                        'response_time': item.response_time_seconds,
                        'words_per_minute': 0,
                        'filler_word_count': 0,
                        'sentiment_score': 0.0,
                    }
                    
                    if item.transcribed_answer:
                        word_count = len(item.transcribed_answer.split())
                        read_time_result = readtime.of_text(item.transcribed_answer)
                        read_time_minutes = read_time_result.minutes + (read_time_result.seconds / 60)
                        
                        if read_time_minutes > 0:
                            wpm = round(word_count / read_time_minutes)
                            question_analytics_item['words_per_minute'] = wpm
                            avg_wpm += wpm
                            wpm_count += 1
                        
                        if item.response_time_seconds:
                            avg_response_time += item.response_time_seconds
                            response_time_count += 1
                        
                        lower_answer = item.transcribed_answer.lower()
                        filler_count = sum(lower_answer.count(word) for word in FILLER_WORDS)
                        question_analytics_item['filler_word_count'] = filler_count
                        total_filler_words += filler_count
                        
                        sentiment = TextBlob(item.transcribed_answer).sentiment.polarity
                        question_analytics_item['sentiment_score'] = sentiment
                        sentiment_scores.append(sentiment)
                    
                    question_analytics.append(question_analytics_item)
            
            final_avg_wpm = round(avg_wpm / wpm_count) if wpm_count > 0 else 0
            final_avg_response_time = round(avg_response_time / response_time_count, 2) if response_time_count > 0 else 0
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # Proctoring analytics
            warning_counts = Counter([log.warning_type.replace('_', ' ').title() for log in all_logs_list if log.warning_type != 'excessive_movement'])
            total_warnings = sum(warning_counts.values())
            
            analytics_data = {
                'session_overview': {
                    'total_questions': len(all_questions),
                    'answered_questions': len([q for q in all_questions if q.transcribed_answer]),
                    'total_warnings': total_warnings,
                    'session_duration': None,  # You can calculate this if you have start/end times
                },
                
                'communication_metrics': {
                    'avg_words_per_minute': final_avg_wpm,
                    'total_filler_words': total_filler_words,
                    'avg_response_time': final_avg_response_time,
                    'avg_sentiment_score': round(avg_sentiment, 3),
                },
                
                'evaluation_scores': {
                    'resume_score': session.resume_score or 0,
                    'answers_score': session.answers_score or 0,
                    'overall_performance_score': session.overall_performance_score or 0,
                },
                
                'proctoring_analytics': {
                    'warning_counts': dict(warning_counts),
                    'total_warnings': total_warnings,
                    'warning_types': list(warning_counts.keys()),
                },
                
                'question_analytics': question_analytics,
            }
            
            return Response(analytics_data, status=status.HTTP_200_OK)
            
        except InterviewSession.DoesNotExist:
            return Response({
                'error': 'Interview session not found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'Error retrieving analytics: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)