# PowerShell script for overnight training on Windows

Write-Host "Starting overnight training run..." -ForegroundColor Green
Write-Host "Logs will be written to: overnight_run.log"
Write-Host "To check progress: Get-Content overnight_run.log -Wait -Tail 20"
Write-Host "To stop: Ctrl+C or close this window"
Write-Host ""

# Change to src directory
Push-Location src

try {
    # Run with output redirection
    python run_overnight.py 2>&1 | Tee-Object -FilePath ..\overnight_run.log
}
finally {
    # Return to original directory
    Pop-Location
}

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Write-Host "View results: cd src; python view_overnight_results.py"
