"""
Simple script to generate an interview link by directly inserting into SQLite database.
Usage: python generate_link_simple.py
"""
import sqlite3
import uuid
from datetime import datetime
import os

# Database path (relative to this script)
# Based on settings.py: BASE_DIR = Path(__file__).resolve().parent.parent
# So if this script is in interview_app/, db.sqlite3 should be in parent/parent (15_11_NEW/)
script_dir = os.path.dirname(os.path.abspath(__file__))  # interview_app/
parent_dir = os.path.dirname(script_dir)  # 15_11_NEW/
db_path = os.path.join(parent_dir, 'db.sqlite3')

# Also check current directory and parent
possible_paths = [
    os.path.join(parent_dir, 'db.sqlite3'),  # 15_11_NEW/db.sqlite3
    os.path.join(script_dir, 'db.sqlite3'),  # interview_app/db.sqlite3
    'db.sqlite3',  # Current working directory
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        break

def generate_link(
    candidate_name='Test Candidate',
    candidate_email='test@example.com',
    job_description='Technical Role',
    resume_text='Experienced professional seeking new opportunities.',
    base_url='http://localhost:8000'
):
    """Generate interview link by directly inserting into database."""
    
    if not db_path or not os.path.exists(db_path):
        print(f"❌ Database not found.")
        print(f"Checked paths: {possible_paths}")
        print("\nTo create the database, run:")
        print("  python manage.py migrate")
        print("\nOr start the Django server and use the /generate-link/ endpoint.")
        print("\nFor now, here's a sample session key and link format:")
        session_key = uuid.uuid4().hex
        sample_link = f"{base_url}/?session_key={session_key}"
        print(f"\nSample Session Key: {session_key}")
        print(f"Sample Link: {sample_link}")
        print("\nYou can manually create the session in the database using this session_key.")
        return None
    
    try:
        # Generate session key
        session_key = uuid.uuid4().hex
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current timestamp
        now = datetime.now().isoformat()
        
        # Insert interview session
        cursor.execute("""
            INSERT INTO interview_app_interviewsession 
            (id, session_key, created_at, candidate_name, candidate_email, 
             job_description, resume_text, scheduled_at, status, language_code, accent_tld)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),  # id
            session_key,        # session_key
            now,                # created_at
            candidate_name,     # candidate_name
            candidate_email,    # candidate_email
            job_description,    # job_description
            resume_text,        # resume_text
            now,                # scheduled_at
            'SCHEDULED',        # status
            'en',               # language_code
            'com'               # accent_tld
        ))
        
        conn.commit()
        conn.close()
        
        # Generate interview link
        interview_link = f"{base_url}/?session_key={session_key}"
        
        print('\n' + '='*70)
        print('✅ Interview Link Generated Successfully!')
        print('='*70)
        print(f'\n📋 Session Details:')
        print(f'   Candidate Name: {candidate_name}')
        print(f'   Candidate Email: {candidate_email}')
        print(f'   Session Key: {session_key}')
        print(f'\n🔗 Interview Link:')
        print(f'   {interview_link}')
        print('='*70 + '\n')
        
        return interview_link
        
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            print(f"❌ Database table not found. Error: {e}")
            print("Please run Django migrations first: python manage.py migrate")
        else:
            print(f"❌ Database error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Generate a link with default values
    link = generate_link(
        candidate_name='Test Candidate',
        candidate_email='test@example.com',
        job_description='Software Engineer Position',
        resume_text='Experienced software engineer with 5+ years of experience.',
        base_url='http://localhost:8000'  # Change this to your actual domain
    )
    
    if link:
        print(f'\n✅ Copy this link to share with the candidate:')
        print(f'{link}\n')

