"""
Standalone script to generate an interview link.
Usage: python generate_interview_link.py

This script can work in two ways:
1. If Django server is running: Makes HTTP request to /generate-link/ endpoint
2. If Django server is not running: Directly creates session in database (requires proper Django setup)
"""
import os
import sys
import requests
import uuid
from datetime import datetime

# Try to use the API endpoint first (if server is running)
def generate_link_via_api(
    candidate_name='Test Candidate',
    candidate_email='test@example.com',
    job_description='Technical Role',
    resume_text='Experienced professional seeking new opportunities.',
    base_url='http://localhost:8000'
):
    """Generate link via HTTP API endpoint."""
    try:
        url = f"{base_url}/generate-link/"
        data = {
            'candidate_name': candidate_name,
            'candidate_email': candidate_email,
            'job_description': job_description,
            'resume_text': resume_text,
        }
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('interview_link')
        print(f"API Error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to server at {base_url}")
        print(f"Error: {e}")
        return None

# Fallback: Direct database access
def generate_link_direct(
    candidate_name='Test Candidate',
    candidate_email='test@example.com',
    job_description='Technical Role',
    resume_text='Experienced professional seeking new opportunities.',
    base_url='http://localhost:8000'
):
    """Generate link by directly accessing the database."""
    try:
        import django
        # Setup Django environment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'interview_app.settings')
        django.setup()
        
        from django.utils import timezone
        from interview_app.models import InterviewSession
        
        session = InterviewSession.objects.create(
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            job_description=job_description,
            resume_text=resume_text,
            scheduled_at=timezone.now(),
            language_code='en',
            accent_tld='com',
            status='SCHEDULED'
        )
        
        interview_link = f"{base_url}/?session_key={session.session_key}"
        return interview_link
    except Exception as e:
        print(f"Direct database access failed: {e}")
        return None


def generate_link(
    candidate_name='Test Candidate',
    candidate_email='test@example.com',
    job_description='Technical Role',
    resume_text='Experienced professional seeking new opportunities.',
    base_url='http://localhost:8000'
):
    """Generate an interview link with the given parameters."""
    
    # Try API first (if server is running)
    print("Attempting to generate link via API...")
    link = generate_link_via_api(
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        job_description=job_description,
        resume_text=resume_text,
        base_url=base_url
    )
    
    if link:
        print('\n' + '='*70)
        print('✅ Interview Link Generated Successfully!')
        print('='*70)
        print(f'\n📋 Session Details:')
        print(f'   Candidate Name: {candidate_name}')
        print(f'   Candidate Email: {candidate_email}')
        print(f'\n🔗 Interview Link:')
        print(f'   {link}')
        print('='*70 + '\n')
        return link
    
    # Fallback to direct database access
    print("\nAPI method failed. Attempting direct database access...")
    link = generate_link_direct(
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        job_description=job_description,
        resume_text=resume_text,
        base_url=base_url
    )
    
    if link:
        print('\n' + '='*70)
        print('✅ Interview Link Generated Successfully!')
        print('='*70)
        print(f'\n📋 Session Details:')
        print(f'   Candidate Name: {candidate_name}')
        print(f'   Candidate Email: {candidate_email}')
        print(f'\n🔗 Interview Link:')
        print(f'   {link}')
        print('='*70 + '\n')
        return link
    
    print('\n❌ Failed to generate interview link using both methods.')
    print('Please ensure:')
    print('1. Django server is running at the base_url, OR')
    print('2. All required Django apps are properly installed')
    return None


if __name__ == '__main__':
    # Generate a link with default values
    # You can modify these values as needed
    link = generate_link(
        candidate_name='Test Candidate',
        candidate_email='test@example.com',
        job_description='Software Engineer Position',
        resume_text='Experienced software engineer with 5+ years of experience.',
        # scheduled_at_str='2024-12-25T14:30',  # Uncomment and set if you want a specific time
        base_url='http://localhost:8000'  # Change this to your actual domain
    )
    
    if link:
        print(f'\n✅ Copy this link to share with the candidate:')
        print(f'{link}\n')

