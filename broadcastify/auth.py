"""
Broadcastify authentication — login and cookie management.
Adapted from github.com/NotJoeMartinez/broadcastify-cli.

Requires a Broadcastify Premium subscription.
Credentials are read from environment variables or a .env file:
  BROADCASTIFY_USERNAME
  BROADCASTIFY_PASSWORD

The session cookie (bcfyuser1) is cached to cookies.json to avoid
re-logging in on every run.
"""

import json
import os
import random
import string
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

LOGIN_URL = "https://www.broadcastify.com/login/"
COOKIES_FILE = Path("cookies.json")

# Mimic a real browser to avoid bot detection
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]


def _random_user_agent() -> str:
    return random.choice(_USER_AGENTS)


def load_cookie() -> str | None:
    """Return cached bcfyuser1 cookie value, or None if not cached."""
    if COOKIES_FILE.exists():
        data = json.loads(COOKIES_FILE.read_text())
        return data.get("bcfyuser1")
    return None


def save_cookie(cookie_value: str) -> None:
    COOKIES_FILE.write_text(json.dumps({"bcfyuser1": cookie_value}))


def login(username: str | None = None, password: str | None = None) -> str:
    """
    Log in to Broadcastify and return the bcfyuser1 session cookie.
    Falls back to BROADCASTIFY_USERNAME / BROADCASTIFY_PASSWORD env vars.
    """
    username = username or os.environ.get("BROADCASTIFY_USERNAME")
    password = password or os.environ.get("BROADCASTIFY_PASSWORD")

    if not username or not password:
        raise ValueError(
            "Broadcastify credentials not found. Set BROADCASTIFY_USERNAME and "
            "BROADCASTIFY_PASSWORD in your environment or .env file."
        )

    session = requests.Session()
    headers = {"user-agent": _random_user_agent()}

    # Fetch login page first to get any CSRF tokens
    session.get(LOGIN_URL, headers=headers)

    resp = session.post(
        LOGIN_URL,
        data={
            "username": username,
            "password": password,
            "action": "auth",
        },
        headers=headers,
        allow_redirects=True,
    )
    resp.raise_for_status()

    cookie_value = session.cookies.get("bcfyuser1")
    if not cookie_value:
        raise RuntimeError(
            "Login failed — bcfyuser1 cookie not returned. "
            "Check your credentials or whether your account has Premium."
        )

    save_cookie(cookie_value)
    return cookie_value


def get_cookie(force_login: bool = False) -> str:
    """
    Return a valid bcfyuser1 cookie, logging in if needed.
    Set force_login=True to bypass the cache and re-authenticate.
    """
    if not force_login:
        cached = load_cookie()
        if cached:
            return cached
    return login()


def auth_headers(cookie: str | None = None) -> dict:
    """Return request headers including auth cookie."""
    cookie = cookie or get_cookie()
    return {
        "user-agent": _random_user_agent(),
        "cookie": f"bcfyuser1={cookie}",
    }
