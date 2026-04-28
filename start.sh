#!/bin/bash
cd "$(dirname "$0")"
rm -rf src/__pycache__
python -m streamlit run app.py
