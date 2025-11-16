import os
from django.core.asgi import get_asgi_application
from django.conf import settings

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import re_path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "interview_app.settings")

django_asgi_app = get_asgi_application()

# Simple Deepgram proxy consumer using websockets
from .deepgram_consumer import deepgram_ws_app  # noqa: E402

websocket_urlpatterns = [
    re_path(r"^dg_ws$", deepgram_ws_app),
]

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": URLRouter(websocket_urlpatterns),
})

