#!/bin/bash

# run.sh - Setup and run the Permittivity Analysis Suite on macOS
# This script checks for uv installation, sets up the environment, and runs the app

set -e  # Exit on any error

echo "🔧 Setting up Permittivity Analysis Suite..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell profile to make uv available in current session
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Verify uv is working
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv installation failed or not in PATH"
    echo "Please restart your terminal and try again, or install uv manually:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "🐍 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    uv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Sync dependencies
echo "📚 Installing dependencies..."
uv sync

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the permittivity_app directory"
    exit 1
fi

echo "🚀 Starting Permittivity Analysis Suite..."
echo "The application will be available at: http://127.0.0.1:8050"
echo "Press Ctrl+C to stop the application"
echo ""

# Run the application
python app.py