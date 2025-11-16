"""
Django management command to generate an interview link.
Usage: python manage.py generate_link [--name NAME] [--email EMAIL] [--job-desc DESC] [--resume-text TEXT]
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from interview_app.models import InterviewSession
import pytz


class Command(BaseCommand):
    help = 'Generate an interview link for a candidate'

    def add_arguments(self, parser):
        parser.add_argument(
            '--name',
            type=str,
            default='Test Candidate',
            help='Candidate name (default: Test Candidate)'
        )
        parser.add_argument(
            '--email',
            type=str,
            default='test@example.com',
            help='Candidate email (default: test@example.com)'
        )
        parser.add_argument(
            '--job-desc',
            type=str,
            default='Technical Role',
            help='Job description (default: Technical Role)'
        )
        parser.add_argument(
            '--resume-text',
            type=str,
            default='Experienced professional seeking new opportunities.',
            help='Resume text (default: Experienced professional...)'
        )
        parser.add_argument(
            '--scheduled-at',
            type=str,
            default=None,
            help='Scheduled time in format YYYY-MM-DDTHH:MM (default: now)'
        )
        parser.add_argument(
            '--language',
            type=str,
            default='en',
            help='Language code (default: en)'
        )
        parser.add_argument(
            '--accent',
            type=str,
            default='com',
            help='Accent TLD (default: com)'
        )
        parser.add_argument(
            '--base-url',
            type=str,
            default='http://localhost:8000',
            help='Base URL for the interview link (default: http://localhost:8000)'
        )

    def handle(self, *args, **options):
        candidate_name = options['name']
        candidate_email = options['email']
        job_description = options['job_desc']
        resume_text = options['resume_text']
        scheduled_at_str = options['scheduled_at']
        language_code = options['language']
        accent_tld = options['accent']
        base_url = options['base_url']

        # Handle scheduled_at
        if scheduled_at_str:
            try:
                ist = pytz.timezone('Asia/Kolkata')
                naive_datetime = timezone.datetime.strptime(scheduled_at_str, '%Y-%m-%dT%H:%M')
                scheduled_at = ist.localize(naive_datetime)
            except (ValueError, pytz.exceptions.InvalidTimeError):
                self.stdout.write(
                    self.style.ERROR('Invalid date format. Use YYYY-MM-DDTHH:MM')
                )
                return
        else:
            scheduled_at = timezone.now()

        # Create interview session
        try:
            session = InterviewSession.objects.create(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_description=job_description,
                resume_text=resume_text,
                scheduled_at=scheduled_at,
                language_code=language_code,
                accent_tld=accent_tld,
                status='SCHEDULED'
            )

            # Generate interview link
            interview_link = f"{base_url}/?session_key={session.session_key}"

            self.stdout.write(self.style.SUCCESS('\n' + '='*70))
            self.stdout.write(self.style.SUCCESS('✅ Interview Link Generated Successfully!'))
            self.stdout.write(self.style.SUCCESS('='*70))
            self.stdout.write(f'\n📋 Session Details:')
            self.stdout.write(f'   Candidate Name: {candidate_name}')
            self.stdout.write(f'   Candidate Email: {candidate_email}')
            self.stdout.write(f'   Session ID: {session.id}')
            self.stdout.write(f'   Session Key: {session.session_key}')
            self.stdout.write(f'   Scheduled At: {scheduled_at}')
            self.stdout.write(f'\n🔗 Interview Link:')
            self.stdout.write(self.style.SUCCESS(f'   {interview_link}'))
            self.stdout.write(self.style.SUCCESS('='*70 + '\n'))

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Failed to create interview session: {str(e)}')
            )

