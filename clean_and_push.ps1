# ===============================================
# HackAudioFeature - Clean and Push Script
# ===============================================

Write-Host "Starting repository cleanup..." -ForegroundColor Green

# Change to project directory
cd "C:\Users\Admin\OneDrive\Desktop\work\HackAudioFeature"

Write-Host "`n[1/6] Checking for large folders..." -ForegroundColor Yellow

# List directories to see what exists
Get-ChildItem -Directory | Format-Table Name

Write-Host "`n[2/6] Moving large folders (venv is already in .gitignore)..." -ForegroundColor Yellow

# venv is already in .gitignore, so we don't need to move it
# Just check what else is there

Write-Host "`n[3/6] Removing old .git history..." -ForegroundColor Yellow

if (Test-Path .git) {
    Remove-Item -Recurse -Force .git
    Write-Host "Removed .git" -ForegroundColor Green
} else {
    Write-Host ".git not found (already removed)" -ForegroundColor Gray
}

Write-Host "`n[4/6] Re-initializing clean git repo..." -ForegroundColor Yellow

git init
git add .
git commit -m "Initial commit: Medical audio monitoring system"
git branch -M main

# Check if remote exists
$remote_exists = git remote | Select-String "origin"

if ($remote_exists) {
    Write-Host "Removing existing remote..." -ForegroundColor Yellow
    git remote remove origin
}

Write-Host "Adding GitHub remote..." -ForegroundColor Yellow
git remote add origin https://github.com/gianmarcomaz/BreathingSoundRecognizer.git

# Verify remote
git remote -v

Write-Host "`n[5/6] Verifying .gitignore exists..." -ForegroundColor Yellow

if (Test-Path .gitignore) {
    Write-Host "✅ .gitignore found" -ForegroundColor Green
    Write-Host "Contents:" -ForegroundColor Gray
    Get-Content .gitignore | Select-Object -First 10
} else {
    Write-Host "Creating .gitignore..." -ForegroundColor Yellow
    @"
# Python
__pycache__/
*.py[cod]
venv/
env/

# Environment variables
.env

# Logs
*.log
logs/

# Audio files
*.wav
*.mp3
*.flac
"@ | Out-File -Encoding utf8 .gitignore
    
    git add .gitignore
    git commit -m "Add .gitignore"
}

Write-Host "`n[6/6] Ready to push!" -ForegroundColor Green
Write-Host "`nRepository status:" -ForegroundColor Cyan
git status

Write-Host "`n" + "="*70 -ForegroundColor Green
Write-Host "READY TO PUSH" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Make sure the repository exists on GitHub first!" -ForegroundColor Yellow
Write-Host ""
Write-Host "If you haven't created it yet:" -ForegroundColor Yellow
Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Create repository: BreathingSoundRecognizer" -ForegroundColor White
Write-Host "3. Leave all checkboxes EMPTY" -ForegroundColor White
Write-Host "4. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "Then run this command to push:" -ForegroundColor Green
Write-Host "   git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Or if you get authentication errors, try:" -ForegroundColor Yellow
Write-Host "   gh auth login" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White
Write-Host ""


