"""
FastAPI dependency injectors (reusable Depends()).
"""

# pyrefly: ignore [missing-import]
from fastapi import Depends, HTTPException, status
# pyrefly: ignore [missing-import]
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.security import verify_token
from app.config import get_settings, Settings

bearer_scheme = HTTPBearer(auto_error=False)


def get_settings_dep() -> Settings:
    return get_settings()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Validate Bearer JWT and return the subject (user ID)."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    subject = verify_token(credentials.credentials)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return subject
