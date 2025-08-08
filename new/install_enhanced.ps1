# Enhanced Installation Script for LLM-Powered Query-Retrieval System v2.0
# PowerShell version for Windows users

Write-Host "🚀 Installing Enhanced LLM Query-Retrieval System v2.0..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Check Python version
$pythonVersion = python --version 2>$null
if ($pythonVersion) {
    Write-Host "📋 Python version: $pythonVersion" -ForegroundColor Blue
} else {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Install core requirements
Write-Host "📦 Installing core Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}

# Check if Redis is requested
$installRedis = Read-Host "🔧 Do you want to install Redis for enhanced caching? (y/n)"

if ($installRedis -eq "y" -or $installRedis -eq "Y") {
    Write-Host "📦 Installing Redis for Windows..." -ForegroundColor Yellow
    
    # Check if Chocolatey is installed
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "📦 Installing Redis via Chocolatey..." -ForegroundColor Blue
        choco install redis-64 -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Redis installed successfully via Chocolatey" -ForegroundColor Green
            Write-Host "🔧 Starting Redis service..." -ForegroundColor Blue
            Start-Service redis
        } else {
            Write-Host "❌ Failed to install Redis via Chocolatey" -ForegroundColor Red
        }
    } else {
        Write-Host "🪟 Chocolatey not found. Alternative Redis options for Windows:" -ForegroundColor Yellow
        Write-Host "   Option 1: Install Chocolatey first: https://chocolatey.org/install" -ForegroundColor White
        Write-Host "   Option 2: Use Docker: docker run -d -p 6379:6379 redis:alpine" -ForegroundColor White
        Write-Host "   Option 3: Use Windows Subsystem for Linux (WSL)" -ForegroundColor White
        Write-Host "   Option 4: Use Redis Cloud (recommended): https://redis.com/try-free/" -ForegroundColor White
    }
} else {
    Write-Host "⏭️  Skipping Redis installation. The system will use file-based caching only." -ForegroundColor Yellow
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "📝 Creating .env file from template..." -ForegroundColor Blue
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "⚠️  Please edit .env file and add your API keys!" -ForegroundColor Yellow
    } else {
        Write-Host "❌ .env.example not found. Creating basic .env file..." -ForegroundColor Red
        @"
# API Keys - Primary
MISTRAL_API_KEY_1=your_primary_mistral_key_here
NOMIC_API_KEY_1=your_primary_nomic_key_here

# API Keys - Backup Keys (Optional)
MISTRAL_API_KEY_2=your_backup_mistral_key_2
NOMIC_API_KEY_2=your_backup_nomic_key_2

# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379

# Deployment Configuration
PORT=8000
"@ | Out-File -FilePath ".env" -Encoding UTF8
    }
} else {
    Write-Host "📝 .env file already exists" -ForegroundColor Blue
}

# Display completion message
Write-Host ""
Write-Host "✅ Installation completed!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "📋 Next steps:" -ForegroundColor Blue
Write-Host "1. Edit .env file and add your API keys:" -ForegroundColor White
Write-Host "   - MISTRAL_API_KEY_1=your_primary_mistral_key" -ForegroundColor Gray
Write-Host "   - NOMIC_API_KEY_1=your_primary_nomic_key" -ForegroundColor Gray

if ($installRedis -eq "y" -or $installRedis -eq "Y") {
    Write-Host "   - REDIS_URL=redis://localhost:6379 (if using local Redis)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "2. Start the application:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test the API:" -ForegroundColor White
Write-Host "   Invoke-WebRequest -Uri http://localhost:8000/ -Method GET" -ForegroundColor Gray
Write-Host ""
Write-Host "🚀 Enhanced features included:" -ForegroundColor Green
Write-Host "   ✅ Multi-layer caching (Memory + Redis + File)" -ForegroundColor White
Write-Host "   ✅ Asynchronous I/O operations" -ForegroundColor White
Write-Host "   ✅ Parallel question processing" -ForegroundColor White
Write-Host "   ✅ Background cache cleanup" -ForegroundColor White
Write-Host "   ✅ Performance monitoring" -ForegroundColor White
Write-Host ""
Write-Host "📚 Read README_v2.md for detailed usage instructions" -ForegroundColor Blue

# Pause to let user read the output
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
