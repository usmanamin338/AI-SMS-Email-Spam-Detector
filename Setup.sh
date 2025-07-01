#!/bin/bash

echo "Setting up your SMS Spam Detector project..."

# Detect OS
OS="$(uname -s)"
IS_WINDOWS=false

case "$OS" in
    CYGWIN*|MINGW*|MSYS*)
        IS_WINDOWS=true
        ;;
esac

# Create virtual environment
if $IS_WINDOWS; then
    python -m venv venv
    echo "To activate the virtual environment on Windows, run:"
    echo "venv\\Scripts\\activate"
else
    python3 -m venv venv
    echo "To activate the virtual environment on Linux/macOS, run:"
    echo "source venv/bin/activate"
fi

# Activate virtual environment for this script session
if $IS_WINDOWS; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
if [ -f nltk.txt ]; then
    while read pkg; do
        python -m nltk.downloader -u -d ./nltk_data "$pkg"
    done < nltk.txt
else
    python -m nltk.downloader -u -d ./nltk_data punkt stopwords
fi

echo "Setup complete!"
echo "You can now run your Streamlit app with:"
echo "streamlit run App.py"
