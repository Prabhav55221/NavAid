#!/bin/bash
# NavAid Demo Launcher

echo "======================================"
echo "  NavAid: Hazard Detection Demo"
echo "======================================"
echo ""

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Warning: GOOGLE_API_KEY environment variable not set"
    echo "You can enter it in the Streamlit sidebar"
    echo ""
fi

# Check if in demo directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run from demo/ directory"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: streamlit not installed"
    echo "Run: pip install -r requirements.txt"
    exit 1
fi

echo "Starting Streamlit app..."
echo ""
echo "Access the app at: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py
