"""Start the Diamond Decision Assistant frontend with: streamlit run app.py."""

import streamlit as st

from src.assistant.client import create_frontend_assistant
from src.assistant.ui import render_app

render_app(st, create_frontend_assistant)
