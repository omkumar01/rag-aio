"""HTTP routers for the web-example app.

Each module owns one page/feature area and exposes a module-level ``router``;
``ROUTERS`` is the registry mounted by :func:`web_example.main.create_app`.
"""

from web_example.auth.routes import router as auth_router
from web_example.routes.chat import router as chat_router
from web_example.routes.documents import router as documents_router
from web_example.routes.landing import router as landing_router
from web_example.routes.settings import router as settings_router
from web_example.routes.upload import router as upload_router

__all__ = ["ROUTERS"]

ROUTERS = [
    landing_router,
    auth_router,
    chat_router,
    documents_router,
    settings_router,
    upload_router,
]
