import uuid
from django.db import models

class InterviewSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=40, unique=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    candidate_name = models.CharField(max_length=100, default="N/A")
    candidate_email = models.EmailField(null=True, blank=True)
    job_description = models.TextField(null=True, blank=True)
    resume_text = models.TextField(null=True, blank=True)
    STATUS_CHOICES = [('SCHEDULED', 'Scheduled'), ('COMPLETED', 'Completed'), ('EXPIRED', 'Expired')]
    scheduled_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='SCHEDULED')
    language_code = models.CharField(max_length=10, default='en')
    accent_tld = models.CharField(max_length=10, default='com')
    is_evaluated = models.BooleanField(default=False)
    resume_summary = models.TextField(null=True, blank=True)
    answers_feedback = models.TextField(null=True, blank=True)
    answers_score = models.FloatField(null=True, blank=True)
    resume_feedback = models.TextField(null=True, blank=True)
    resume_score = models.FloatField(null=True, blank=True)
    keyword_analysis = models.TextField(null=True, blank=True)
    overall_performance_feedback = models.TextField(null=True, blank=True)
    overall_performance_score = models.FloatField(null=True, blank=True)
    behavioral_analysis = models.TextField(null=True, blank=True)
    id_verification_status = models.CharField(max_length=50, default='Pending')
    id_card_image = models.ImageField(upload_to='id_cards/', null=True, blank=True)
    extracted_id_details = models.TextField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.session_key:
            self.session_key = uuid.uuid4().hex
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Interview for {self.candidate_name} on {self.created_at.strftime('%Y-%m-%d')}"

class WarningLog(models.Model):
    session = models.ForeignKey(InterviewSession, related_name='logs', on_delete=models.CASCADE)
    warning_type = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    snapshot = models.CharField(max_length=255, null=True, blank=True, help_text="Filename of the snapshot image captured when warning occurred")

    def __str__(self):
        return f"Warning ({self.warning_type}) for {self.session.candidate_name}"

class InterviewQuestion(models.Model):
    session = models.ForeignKey(InterviewSession, related_name='questions', on_delete=models.CASCADE)
    question_text = models.TextField()
    
    QUESTION_TYPE_CHOICES = [
        ('TECHNICAL', 'Technical'),
        ('BEHAVIORAL', 'Behavioral'),
    ]
    question_type = models.CharField(max_length=50, choices=QUESTION_TYPE_CHOICES, default='TECHNICAL')
    
    audio_url = models.URLField(max_length=500, null=True, blank=True)
    question_level = models.CharField(max_length=10, default='MAIN')
    parent_question = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='follow_ups')
    transcribed_answer = models.TextField(null=True, blank=True)
    order = models.PositiveIntegerField()
    words_per_minute = models.IntegerField(null=True, blank=True)
    filler_word_count = models.IntegerField(null=True, blank=True)
    response_time_seconds = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Q{self.order + 1} ({self.question_level}) for {self.session.candidate_name}"