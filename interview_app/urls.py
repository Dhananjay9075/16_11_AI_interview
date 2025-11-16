from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # Existing URLs for the app
    path('start/', views.start_interview, name='start_interview'),
    path('invite/', views.create_interview_invite, name='create_invite'),
    path('generate-link/', views.generate_interview_link, name='generate_interview_link'),
    path('', views.interview_portal, name='interview_portal'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('report/<uuid:session_id>/pdf/', views.download_report_pdf, name='download_report_pdf'),
    path('report/<uuid:session_id>/proctoring_pdf/', views.download_proctoring_pdf, name='download_proctoring_pdf'),
    path('report/<uuid:session_id>/', views.interview_report, name='interview_report'),
    path('complete/', views.interview_complete, name='interview_complete'),

    # API Endpoints
    path('video_feed/', views.video_feed, name='video_feed'),
    path('video_frame/', views.video_frame, name='video_frame'),
    path('status/', views.get_proctoring_status, name='get_proctoring_status'),
    path('report_tab_switch/', views.report_tab_switch, name='report_tab_switch'),
    path('transcribe/', views.transcribe_audio, name='transcribe_audio'),
    path('check_camera/', views.check_camera, name='check_camera'),
    path('activate_proctoring/', views.activate_proctoring_camera, name='activate_proctoring_camera'),
    path('end_session/', views.end_interview_session, name='end_interview_session'),
    path('release_camera/', views.release_camera, name='release_camera'),
    path('execute_code/', views.execute_code, name='execute_code'),
    path('submit_coding_challenge/', views.submit_coding_challenge, name='submit_coding_challenge'),
    path('verify_id/', views.verify_id, name='verify_id'),
    
    # --- NEW API ENDPOINTS FOR INTERVIEW RESULTS ---
    path('api/results/<uuid:session_id>/', views.InterviewResultsAPIView.as_view(), name='interview_results_api'),
    path('api/results/', views.InterviewResultsListAPIView.as_view(), name='interview_results_list_api'),
    path('api/analytics/<uuid:session_id>/', views.InterviewAnalyticsAPIView.as_view(), name='interview_analytics_api'),

    # AI Chatbot endpoints
    path('chatbot/', views.chatbot_standalone, name='chatbot_standalone'),
    path('ai/start', views.ai_start, name='ai_start'),
    path('ai/upload_answer', views.ai_upload_answer, name='ai_upload_answer'),
    path('ai/repeat', views.ai_repeat, name='ai_repeat'),
    path('ai/transcript_pdf', views.ai_transcript_pdf, name='ai_transcript_pdf'),

    # The following API includes exist in the full project but the apps are
    # not present in this minimal 15_11_NEW copy. They are commented out so
    # that core interview features and management commands can run.
    #
    # # Auth endpoints namespace
    # path('api/auth/', include('authapp.urls')),
    #
    # # Include app APIs under /api/*
    # path('api/jobs/', include('jobs.urls')),
    # path('api/candidates/', include('candidates.urls')),
    # path('api/companies/', include('companies.urls')),
    # path('api/dashboard/', include('dashboard.urls')),
    # path('api/interviews/', include('interviews.urls')),
    # path('api/evaluation/', include('evaluation.urls')),  # Evaluation CRUD endpoints
    # path('api/requests/', include('candidates.urls')),  # Requests endpoints (pending, etc.)
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)