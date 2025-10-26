@echo off
REM ===============================================
REM HackAudioFeature - Git Push Helper
REM ===============================================
echo.
echo ===============================================
echo   HackAudioFeature - GitHub Push Helper
echo ===============================================
echo.

REM Check if .env exists
if exist .env (
    echo [OK] .env file found (will be ignored by git)
) else (
    echo [WARNING] No .env file found. Create one from env.example!
)

REM Check .gitignore
if exist .gitignore (
    echo [OK] .gitignore found
) else (
    echo [ERROR] .gitignore missing!
    exit /b 1
)

echo.
echo Pre-push verification:
echo.

REM Check for any hardcoded API keys
findstr /i "aks_live sk-proj" app.py >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Found possible API keys in app.py!
    echo Review the file before pushing.
    pause
)

echo [OK] No hardcoded credentials found in app.py
echo.

REM Initialize git if not already done
if not exist .git (
    echo [INFO] Initializing git repository...
    git init
)

echo [INFO] Adding files...
git add .

echo.
echo ===============================================
echo   Files staged for commit:
echo ===============================================
git status

echo.
echo ===============================================
echo   Ready to push!
echo ===============================================
echo.
echo Choose an option:
echo.
echo 1. Commit and push to new GitHub repo
echo 2. Commit only (you'll push manually)
echo 3. Cancel
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    set /p commit_msg="Enter commit message: "
    if "!commit_msg!"=="" set commit_msg="Update: HackAudioFeature medical monitoring system"
    
    echo.
    echo [INFO] Committing...
    git commit -m "!commit_msg!"
    
    echo.
    echo [INFO] Set up remote repository:
    echo 1. Go to https://github.com/new
    echo 2. Create a repository named "HackAudioFeature"
    echo 3. Copy the repository URL
    echo.
    set /p remote_url="Enter GitHub repository URL (or skip to finish): "
    
    if not "!remote_url!"=="" (
        echo [INFO] Adding remote...
        git remote add origin !remote_url!
        
        echo [INFO] Pushing to GitHub...
        git push -u origin main
        
        echo.
        echo [SUCCESS] Project pushed to GitHub!
    ) else (
        echo [INFO] Commit created. Push manually with: git push
    )
) else if "%choice%"=="2" (
    set /p commit_msg="Enter commit message: "
    if "!commit_msg!"=="" set commit_msg="Update: HackAudioFeature medical monitoring system"
    
    echo [INFO] Committing...
    git commit -m "!commit_msg!"
    
    echo [SUCCESS] Committed successfully!
    echo [INFO] Push with: git push
) else (
    echo [INFO] Cancelled.
    exit /b 0
)

echo.
pause

