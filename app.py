"""Start the local Diamond Decision Assistant with: python -m streamlit run app.py."""

import streamlit as st

from src.assistant.runtime import create_assistant
from src.assistant.ui import render_app

render_app(st, create_assistant)
