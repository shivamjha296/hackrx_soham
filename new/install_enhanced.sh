#!/bin/bash

# Enhanced Installation Script for LLM-Powered Query-Retrieval System v2.0
# This script installs all required dependencies including optional Redis support

echo "🚀 Installing Enhanced LLM Query-Retrieval System v2.0..."
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
echo "📋 Python version: $python_version"

# Install core requirements
echo "📦 Installing core Python dependencies..."
pip install -r requirements.txt

# Check if Redis is requested
read -p "🔧 Do you want to install Redis for enhanced caching? (y/n): " install_redis

if [[ $install_redis == "y" || $install_redis == "Y" ]]; then
    echo "📦 Installing Redis..."
    
    # Detect OS and install Redis
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "🐧 Detected Linux - Installing Redis via apt..."
        sudo apt-get update
        sudo apt-get install -y redis-server
        sudo systemctl start redis
        sudo systemctl enable redis
        echo "✅ Redis installed and started on Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        echo "🍎 Detected macOS - Installing Redis via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install redis
            brew services start redis
            echo "✅ Redis installed and started on macOS"
        else
            echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        echo "🪟 Detected Windows - Please install Redis manually:"
        echo "   Option 1: Use Windows Subsystem for Linux (WSL)"
        echo "   Option 2: Use Docker: docker run -d -p 6379:6379 redis:alpine"
        echo "   Option 3: Use Redis Cloud (recommended): https://redis.com/try-free/"
    else
        echo "❓ Unknown OS. Please install Redis manually."
    fi
else
    echo "⏭️  Skipping Redis installation. The system will use file-based caching only."
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys!"
else
    echo "📝 .env file already exists"
fi

# Display completion message
echo ""
echo "✅ Installation completed!"
echo "=================================================="
echo "📋 Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - MISTRAL_API_KEY_1=your_primary_mistral_key"
echo "   - NOMIC_API_KEY_1=your_primary_nomic_key"
if [[ $install_redis == "y" || $install_redis == "Y" ]]; then
    echo "   - REDIS_URL=redis://localhost:6379 (if using local Redis)"
fi
echo ""
echo "2. Start the application:"
echo "   python main.py"
echo ""
echo "3. Test the API:"
echo "   curl -X GET http://localhost:8000/"
echo ""
echo "🚀 Enhanced features included:"
echo "   ✅ Multi-layer caching (Memory + Redis + File)"
echo "   ✅ Asynchronous I/O operations"
echo "   ✅ Parallel question processing"
echo "   ✅ Background cache cleanup"
echo "   ✅ Performance monitoring"
echo ""
echo "📚 Read README_v2.md for detailed usage instructions"
