"""Gunicorn entrypoint: `gunicorn wsgi:app`."""
from src.api.app import create_app

app = create_app()
