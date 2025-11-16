# AI-Powered Interview Platform

An intelligent interview platform with AI-powered question generation, real-time proctoring, speech transcription, and comprehensive candidate evaluation.

## Features

- 🤖 **AI-Powered Interviews**: Automated question generation using Google Gemini AI
- 🎥 **Real-time Proctoring**: Camera-based monitoring with face detection using YOLO
- 🎤 **Speech Transcription**: Real-time audio transcription using OpenAI Whisper and Deepgram
- 📝 **Resume Analysis**: Automatic resume parsing and evaluation
- 📊 **Comprehensive Reports**: Detailed PDF reports with candidate performance metrics
- 🔐 **ID Verification**: Document verification system
- 💬 **Interactive Chatbot**: AI chatbot for interview assistance

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Web UI     │  │   Camera     │  │   Microphone │           │
│  │  (Portal)    │  │  (Webcam)    │  │   (Audio)    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                    │
│                            │                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             │ HTTP/WebSocket
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    DJANGO BACKEND SERVER                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Django Framework                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │   Views.py   │  │   URLs.py    │  │  Models.py    │     │  │
│  │  │  (API/Logic) │  │  (Routing)   │  │  (Database)   │     │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │  │
│  └─────────┼─────────────────┼─────────────────┼──────────────┘  │
│            │                  │                  │                  │
│  ┌─────────▼──────────────────▼──────────────────▼──────────────┐  │
│  │                    Core Modules                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │ AI Chatbot   │  │   Camera     │  │  Proctoring   │     │  │
│  │  │  Manager     │  │   Handler    │  │   Monitor     │     │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │  │
│  └─────────┼─────────────────┼─────────────────┼──────────────┘  │
│            │                  │                  │                  │
└────────────┼──────────────────┼──────────────────┼─────────────────┘
             │                  │                  │
             │                  │                  │
    ┌────────▼────────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │  AI Services     │  │  ML Models   │  │  Storage    │
    │                  │  │              │  │             │
    │  ┌────────────┐ │  │  ┌────────┐ │  │  ┌────────┐│
    │  │  Gemini AI │ │  │  │  YOLO  │ │  │  │ SQLite  ││
    │  │  (Q&A Gen) │ │  │  │  (Face) │ │  │  │   DB    ││
    │  └────────────┘ │  │  └────────┘ │  │  └────────┘│
    │  ┌────────────┐ │  │  ┌────────┐ │  │  ┌────────┐│
    │  │ Google TTS │ │  │  │  │       │  │    Media  ││
    │  │  (Speech)  │ │  │  │  │       │  │    Files  ││
    │  └────────────┘ │  │  └────────┘ │  │  └────────┘│
    │  ┌────────────┐ │  │  ┌────────┐ │  │             │
    │  │  Deepgram   │ │  │  │ OpenCV │ │  │             │
    │  │(Transcribe) │ │  │  │(Video) │ │  │             │
    │  └────────────┘ │  │  └────────┘ │  │             │
    └──────────────────┘  └─────────────┘  └─────────────┘
```


## Interview Flow Diagram

### Complete Interview Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                    INTERVIEW LIFECYCLE                            │
└──────────────────────────────────────────────────────────────────┘

    [1] SESSION CREATION
         │
         ├─► Admin/HR creates interview session
         ├─► Candidate details entered (Name, Email, Resume, JD)
         ├─► System generates unique session_key
         └─► Interview link created: /?session_key=abc123...
         │
         ▼
    [2] CANDIDATE ACCESS
         │
         ├─► Candidate opens interview link
         ├─► System validates session_key
         └─► Portal loads with session context
         │
         ▼
    [3] ID VERIFICATION
         │
         ├─► Candidate uploads ID card image
         ├─► System extracts ID details (OCR/AI)
         ├─► Verification status stored
         └─► Proceed to camera check
         │
         ▼
    [4] CAMERA & AUDIO SETUP
         │
         ├─► Browser requests camera/microphone access
         ├─► System verifies device availability
         ├─► Camera feed initialized
         └─► Audio recording ready
         │
         ▼
    [5] PROCTORING ACTIVATION
         │
         ├─► Real-time video capture starts
         ├─► YOLO face detection initialized
         ├─► Proctoring monitoring begins
         └─► Warning system active
         │
         ▼
    [6] AI INTERVIEW START
         │
         ├─► Resume text analyzed
         ├─► Job description processed
         ├─► Gemini AI generates first question
         ├─► Question converted to speech (Google TTS)
         └─► Audio URL returned to frontend
         │
         ▼
    ┌──────────────────────────────────────────────────────────┐
    │              QUESTION-ANSWER LOOP (Repeats)               │
    └──────────────────────────────────────────────────────────┘
         │
         ├─► [7] QUESTION PRESENTATION
         │    │
         │    ├─► Question text displayed
         │    ├─► Audio playback (if available)
         │    └─► Timer starts
         │    │
         │    ▼
         ├─► [8] CANDIDATE RESPONSE
         │    │
         │    ├─► Candidate speaks answer
         │    ├─► Audio recorded in real-time
         │    ├─► Deepgram transcribes speech
         │    └─► Transcript stored
         │    │
         │    ▼
         ├─► [9] PROCTORING MONITORING (Continuous)
         │    │
         │    ├─► Face detection (YOLO)
         │    ├─► Multiple person detection
         │    ├─► Tab switching detection
         │    ├─► Warning logs created
         │    └─► Snapshots captured on warnings
         │    │
         │    ▼
         ├─► [10] AI EVALUATION
         │    │
         │    ├─► Gemini AI analyzes answer
         │    ├─► Scores assigned (content, clarity, etc.)
         │    ├─► Feedback generated
         │    
         │    │
         │    ▼
         └─► [11] NEXT QUESTION GENERATION
              │
              ├─► AI decides next question type
              ├─► Context from previous answers used
              ├─► Question generated
              └─► Loop continues until max questions reached
              │
              ▼
    [12] INTERVIEW COMPLETION
         │
         ├─► Final evaluation performed
         ├─► Overall scores calculated
         ├─► Behavioral analysis generated
         └─► Session status → COMPLETED
         │
         ▼
    [13] REPORT GENERATION
         │
         ├─► Comprehensive report created
         ├─► PDF generated (WeasyPrint)
         ├─► Includes:
         │   ├─► Candidate performance scores
         │   ├─► Question-answer pairs
         │   ├─► AI feedback
         │   ├─► Proctoring warnings
         │   └─► Recommendations
         └─► Report downloadable
         │
         ▼
    [14] SESSION CLOSURE
         │
         ├─► Camera resources released
         ├─► Session archived
         └─► Data available for review
```



## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 15_11_NEW
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter dependency conflicts, try:
```bash
pip install -r requirements.txt --legacy-peer-deps
```

### 4. Environment Configuration

Create a `.env` file in the project root directory:

```env 
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=1

# Gemini AI API Key (Required for AI features)
GEMINI_API_KEY=your-gemini-api-key-here

# Google Cloud Text-to-Speech (Optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json

# Deepgram API Key (Optional, for real-time transcription)
DEEPGRAM_API_KEY=your-deepgram-api-key-here


### 5. Database Setup

Run migrations to create the database:

```bash
python manage.py migrate
```

Create a superuser (optional, for admin access):

```bash
python manage.py createsuperuser
```

## Running the Server

### Start the Development Server

```bash
python manage.py runserver
```

The server will start at `http://localhost:8000`

### Access Points

- **Interview Portal**: `http://localhost:8000/`

## Starting an Interview

There are multiple ways to start an interview session:

### Method 1: Using the Web Interface

1. Navigate to `http://localhost:8000/start/`
2. Fill in the form:
   - Candidate Name (required)
   - Candidate Email (optional)
   - Job Description (optional)

3. Click "Start Interview"
4. You'll be redirected to the interview portal with the session key



## Interview Flow

1. **Access Interview Link**: Candidate opens the interview link
2. **ID Verification**: Candidate uploads ID card for verification
3. **Camera Check**: System verifies camera access
4. **Interview Start**: AI generates questions based on resume and job description
5. **Real-time Proctoring**: System monitors candidate behavior
6. **Question-Answer**: Candidate answers AI-generated questions
7. **Evaluation**: AI evaluates responses and generates feedback
8. **Report Generation**: Comprehensive PDF report is generated



