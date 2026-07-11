# models.py
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from typing import Optional
import re


class UserCreate(BaseModel):
    """Validated payload for signup."""
    username: str
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if len(v) > 30:
            raise ValueError("Username must be 30 characters or fewer.")
        if not re.match(r'^[A-Za-z0-9_]+$', v):
            raise ValueError("Username may only contain letters, digits, and underscores.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strong(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one digit.")
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "UserCreate":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class UserLogin(BaseModel):
    """Validated payload for login."""
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().lower()


class UserInDB(BaseModel):
    """User record as stored — password is bcrypt hash, never plaintext."""
    user_id: int
    username: str
    email: str
    hashed_password: str
    is_active: bool = True


class UserPublic(BaseModel):
    """Safe user info returned to the UI — no password hash."""
    user_id: int
    username: str
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Claims decoded from a JWT."""
    user_id: int
    username: str