"""
Comprehensive PDF generation including Q&A and Coding Round results
"""
try:
    from fpdf import FPDF
except ImportError:
    try:
        from fpdf2 import FPDF
    except ImportError:
        FPDF = None

from .models import InterviewSession, CodeSubmission


def _sanitize_for_pdf(text: str) -> str:
    """Sanitize text for PDF encoding"""
    try:
        return text.encode('latin-1', 'replace').decode('latin-1')
    except Exception:
        return text


def _wrap_long_words(text: str, max_len: int = 50) -> str:
    """Wrap long words to prevent PDF overflow - more aggressive wrapping"""
    if not text:
        return ""
    
    parts = []
    for token in text.split(' '):
        if len(token) <= max_len:
            parts.append(token)
        else:
            # Break very long tokens into smaller chunks
            for i in range(0, len(token), max_len):
                chunk = token[i:i+max_len]
                parts.append(chunk)
    
    result = ' '.join(parts)
    
    # Additional safety: if result is still too long, truncate it
    if len(result) > 10000:
        result = result[:10000] + "... (truncated)"
    
    return result


def generate_comprehensive_pdf(session_key: str) -> bytes:
    """
    Generate comprehensive PDF including:
    - Technical Q&A transcript
    - Coding challenge results
    - AI evaluation
    """
    if FPDF is None:
        print("❌ FPDF not available")
        return b""
    
    try:
        # Get session from database
        try:
            session = InterviewSession.objects.get(session_key=session_key)
        except InterviewSession.DoesNotExist:
            print(f"❌ Session not found: {session_key}")
            return b""
        
        # Get AI chatbot session for Q&A transcript - try multiple lookup methods
        ai_session = None
        
        # Try 1: Look up by Django session ID
        from .ai_chatbot import chatbot_manager
        ai_session = chatbot_manager.sessions.get(str(session.id))
        
        if not ai_session:
            # Try 2: Look up in complete_ai_bot by session ID
            from .complete_ai_bot import sessions as complete_sessions
            ai_session = complete_sessions.get(str(session.id))
        
        if not ai_session:
            # Try 3: Search all sessions in complete_ai_bot for matching candidate
            from .complete_ai_bot import sessions as complete_sessions
            for sid, sess in complete_sessions.items():
                if sess.candidate_name == session.candidate_name:
                    ai_session = sess
                    print(f"✅ Found AI session by candidate name match: {sid}")
                    break
        
        if not ai_session:
            print(f"⚠️ No AI conversation history found for session {session.id}")
        
        # Get coding submissions
        coding_submissions = CodeSubmission.objects.filter(session=session).order_by('created_at')
        
        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        
        # === HEADER ===
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, _sanitize_for_pdf("AI Interview Report"), ln=True, align='C')
        pdf.ln(5)
        
        # === CANDIDATE INFO ===
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, _sanitize_for_pdf("Candidate Information"), ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 6, _sanitize_for_pdf(f"Name: {session.candidate_name}"), ln=True)
        pdf.cell(0, 6, _sanitize_for_pdf(f"Email: {session.candidate_email or 'N/A'}"), ln=True)
        pdf.cell(0, 6, _sanitize_for_pdf(f"Date: {session.created_at.strftime('%Y-%m-%d %H:%M')}"), ln=True)
        pdf.cell(0, 6, _sanitize_for_pdf(f"Session ID: {session.id}"), ln=True)
        pdf.ln(8)
        
        # === TECHNICAL Q&A SECTION ===
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, _sanitize_for_pdf("Technical Q&A Transcript"), ln=True)
        pdf.ln(3)
        
        if ai_session and ai_session.conversation_history:
            pdf.set_font("Arial", size=11)
            for msg in ai_session.conversation_history:
                role = "Interviewer" if msg.get("role") == "interviewer" else "Candidate"
                text = msg.get("text", "")
                
                # Role header
                pdf.set_font("Arial", "B", 11)
                try:
                    pdf.multi_cell(0, 6, _sanitize_for_pdf(role + ":"), align='L')
                except:
                    pdf.cell(0, 6, _sanitize_for_pdf(role + ":"), ln=True)
                
                # Message text
                pdf.set_font("Arial", size=10)  # Slightly smaller for better fit
                safe_text = _sanitize_for_pdf(_wrap_long_words(text, max_len=60))
                try:
                    pdf.multi_cell(0, 5, safe_text, align='L')
                except Exception as e:
                    # Fallback: truncate and use cell instead
                    truncated = safe_text[:200] + "..." if len(safe_text) > 200 else safe_text
                    pdf.cell(0, 5, truncated, ln=True)
                pdf.ln(2)
        else:
            pdf.set_font("Arial", "I", 11)
            pdf.cell(0, 6, _sanitize_for_pdf("No Q&A transcript available"), ln=True)
        
        pdf.ln(5)
        
        # === CODING ROUND SECTION ===
        if coding_submissions.exists():
            pdf.add_page()  # New page for coding
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, _sanitize_for_pdf("Coding Challenge Results"), ln=True)
            pdf.ln(3)
            
            for idx, submission in enumerate(coding_submissions, 1):
                # Get question
                from .models import InterviewQuestion
                try:
                    question = InterviewQuestion.objects.get(id=submission.question_id)
                    question_text = question.question_text
                except:
                    question_text = "Coding Challenge"
                
                # Challenge header
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, _sanitize_for_pdf(f"Challenge {idx}: {question_text[:60]}"), ln=True)
                pdf.ln(2)
                
                # Language and status
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 6, _sanitize_for_pdf(f"Language: {submission.language}"), ln=True)
                status_text = "All Tests Passed" if submission.passed_all_tests else "Some Tests Failed"
                pdf.cell(0, 6, _sanitize_for_pdf(f"Status: {status_text}"), ln=True)
                pdf.ln(2)
                
                # Gemini evaluation if available
                if submission.gemini_evaluation:
                    gemini_score = submission.gemini_evaluation.get('score', 'N/A')
                    passed_tests = submission.gemini_evaluation.get('passed_tests', 0)
                    total_tests = submission.gemini_evaluation.get('total_tests', 0)
                    
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, _sanitize_for_pdf(f"AI Score: {gemini_score}/100"), ln=True)
                    pdf.cell(0, 6, _sanitize_for_pdf(f"Tests Passed: {passed_tests}/{total_tests}"), ln=True)
                    pdf.ln(2)
                
                # Submitted code
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, _sanitize_for_pdf("Submitted Code:"), ln=True)
                pdf.set_font("Courier", size=8)  # Smaller font for code
                code_lines = submission.submitted_code.split('\n')[:30]  # Limit to 30 lines
                for line in code_lines:
                    safe_line = _sanitize_for_pdf(_wrap_long_words(line, max_len=80))  # Wrap long lines
                    try:
                        pdf.multi_cell(0, 4, safe_line, align='L')
                    except Exception as e:
                        # Skip lines that are too long to render
                        pdf.set_font("Arial", "I", 8)
                        pdf.cell(0, 4, _sanitize_for_pdf("... (line too long to display)"), ln=True)
                        pdf.set_font("Courier", size=8)
                
                if len(submission.submitted_code.split('\n')) > 30:
                    pdf.set_font("Arial", "I", 9)
                    pdf.cell(0, 5, _sanitize_for_pdf("... (code truncated)"), ln=True)
                
                pdf.ln(3)
                
                # Test results / Output log
                if submission.output_log:
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 6, _sanitize_for_pdf("Test Results:"), ln=True)
                    pdf.set_font("Arial", size=9)
                    log_lines = submission.output_log.split('\n')[:20]  # Limit output
                    for line in log_lines:
                        safe_line = _sanitize_for_pdf(_wrap_long_words(line, max_len=70))
                        try:
                            pdf.multi_cell(0, 5, safe_line, align='L')
                        except Exception as e:
                            # Skip lines that cause rendering errors
                            pdf.cell(0, 5, _sanitize_for_pdf("... (line skipped)"), ln=True)
                
                pdf.ln(5)
        
        # === GEMINI AI ANALYSIS ===
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, _sanitize_for_pdf("AI Analysis & Evaluation"), ln=True)
        pdf.ln(3)
        
        # Generate Gemini analysis for technical Q&A
        if ai_session and ai_session.conversation_history:
            try:
                import google.generativeai as genai
                from django.conf import settings
                api_key = getattr(settings, 'GEMINI_API_KEY', '')
                if api_key:
                    genai.configure(api_key=api_key)
                else:
                    raise ValueError("GEMINI_API_KEY not set in environment")
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                # Build conversation context
                conversation_text = "\n".join([
                    f"{msg.get('role', 'unknown').upper()}: {msg.get('text', '')}"
                    for msg in ai_session.conversation_history
                ])
                
                technical_prompt = f"""
                Analyze this technical interview transcript and provide a comprehensive evaluation.
                
                Job Description: {session.job_description[:500] if session.job_description else 'AI/ML Intern'}
                Candidate: {session.candidate_name}
                
                Interview Transcript:
                {conversation_text[:3000]}
                
                Provide:
                1. Overall technical competency score (0-100)
                2. Key strengths identified
                3. Areas for improvement
                4. Technical knowledge assessment
                5. Communication skills evaluation
                6. Recommendation (Strong/Moderate/Weak candidate)
                
                Format as structured text.
                """
                
                technical_response = model.generate_content(technical_prompt)
                technical_analysis = technical_response.text if technical_response.text else "Analysis unavailable"
                
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, _sanitize_for_pdf("Technical Interview Analysis"), ln=True)
                pdf.ln(2)
                pdf.set_font("Arial", size=10)
                analysis_text = _sanitize_for_pdf(_wrap_long_words(technical_analysis, max_len=70))
                try:
                    pdf.multi_cell(0, 5, analysis_text, align='L')
                except:
                    pdf.cell(0, 5, _sanitize_for_pdf(technical_analysis[:500] + "..."), ln=True)
                pdf.ln(5)
            except Exception as e:
                print(f"⚠️ Error generating technical analysis: {e}")
                pdf.set_font("Arial", "I", 11)
                pdf.cell(0, 6, _sanitize_for_pdf("Technical analysis unavailable"), ln=True)
        
        # Generate Gemini analysis for coding round
        if coding_submissions.exists():
            try:
                import google.generativeai as genai
                from django.conf import settings
                api_key = getattr(settings, 'GEMINI_API_KEY', '')
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                else:
                    raise ValueError("GEMINI_API_KEY not set in environment")
                
                coding_analysis_text = []
                for submission in coding_submissions:
                    coding_analysis_text.append(f"""
                    Language: {submission.language}
                    Code: {submission.submitted_code[:1000]}
                    Tests Passed: {'Yes' if submission.passed_all_tests else 'No'}
                    Output: {submission.output_log[:500] if submission.output_log else 'N/A'}
                    """)
                
                coding_prompt = f"""
                Analyze this coding challenge submission and provide evaluation.
                
                Submissions:
                {''.join(coding_analysis_text)}
                
                Provide:
                1. Code quality score (0-100)
                2. Problem-solving approach assessment
                3. Code correctness and efficiency
                4. Best practices adherence
                5. Suggestions for improvement
                6. Overall coding competency rating
                
                Format as structured text.
                """
                
                coding_response = model.generate_content(coding_prompt)
                coding_analysis = coding_response.text if coding_response.text else "Analysis unavailable"
                
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, _sanitize_for_pdf("Coding Challenge Analysis"), ln=True)
                pdf.ln(2)
                pdf.set_font("Arial", size=10)
                analysis_text = _sanitize_for_pdf(_wrap_long_words(coding_analysis, max_len=70))
                try:
                    pdf.multi_cell(0, 5, analysis_text, align='L')
                except:
                    pdf.cell(0, 5, _sanitize_for_pdf(coding_analysis[:500] + "..."), ln=True)
            except Exception as e:
                print(f"⚠️ Error generating coding analysis: {e}")
                pdf.set_font("Arial", "I", 11)
                pdf.cell(0, 6, _sanitize_for_pdf("Coding analysis unavailable"), ln=True)
        
        # === OVERALL EVALUATION ===
        if session.overall_performance_feedback or session.overall_performance_score:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, _sanitize_for_pdf("Overall Performance Summary"), ln=True)
            pdf.ln(2)
            
            if session.overall_performance_score:
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, _sanitize_for_pdf(f"Overall Score: {session.overall_performance_score:.1f}/100"), ln=True)
                pdf.ln(2)
            
            if session.overall_performance_feedback:
                pdf.set_font("Arial", size=10)
                feedback_text = _sanitize_for_pdf(_wrap_long_words(session.overall_performance_feedback))
                try:
                    pdf.multi_cell(0, 6, feedback_text, align='L')
                except:
                    pdf.cell(0, 6, _sanitize_for_pdf(feedback_text[:300] + "..."), ln=True)
        
        # Generate PDF bytes
        data = pdf.output(dest='S')
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return str(data).encode('latin1', 'replace')
        
    except Exception as e:
        print(f"❌ Error generating comprehensive PDF: {e}")
        import traceback
        traceback.print_exc()
        return b""


def ai_comprehensive_pdf_django(session_key: str) -> bytes:
    """
    Generate comprehensive PDF for a session (wrapper for views.py)
    """
    return generate_comprehensive_pdf(session_key)

