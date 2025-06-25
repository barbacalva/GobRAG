from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from gobrag.config import settings


def attach_middlewares(app):
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # Rate-limit
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Auth + rate-limit
    @app.middleware("http")
    async def api_key_and_rate_limit(request: Request, call_next):
        if request.url.path in {"/health", "/ready"}:
            return await call_next(request)

        key = request.headers.get("x-api-key")
        if key not in settings.api_keys:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid API key"}
            )

        request.state.api_key = key
        limit_decorator = limiter.limit(settings.rate_limit)
        try:
            return await limit_decorator(call_next)(request)
        except RateLimitExceeded:
            return JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"}
            )
