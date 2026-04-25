#!/bin/bash
cd "$(dirname "$0")"
rm -rf src/__pycache__
/Users/knight/ai-agent-projects/venv/bin/streamlit run app.py
