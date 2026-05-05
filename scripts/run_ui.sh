#!/bin/bash
# Run the Resume Stress Test UI
#
# Prerequisites:
#   export GOOGLE_API_KEY="your-google-api-key"
#   export OPENROUTER_API_KEY="your-openrouter-api-key"
#
# Usage:
#   ./scripts/run_ui.sh
#
# The app will be available at http://localhost:8501

set -x

cd "$(dirname "$0")/.."

# Check for required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Warning: GOOGLE_API_KEY not set. Resume processing will fail."
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. Model evaluation will fail."
fi

# Run Streamlit app (using ui_draft)
streamlit run ui/Resume_Screening_Validity.py --server.port 8501 --server.address 0.0.0.0

