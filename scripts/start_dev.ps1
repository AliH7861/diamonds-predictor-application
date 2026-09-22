[CmdletBinding()]
param(
    [int]$BackendPort = 8770,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogRoot = Join-Path $ProjectRoot "outputs\logs"
$ProcessFile = Join-Path $LogRoot "dev-processes.json"
$BackendUrl = "http://localhost:$BackendPort"
$FrontendUrl = "http://localhost:$FrontendPort"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment not found at $PythonPath. Create .venv and install requirements first."
}

$NpmPath = (Get-Command npm.cmd -ErrorAction Stop).Source

function Test-Url {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$TimeoutSeconds,
        [string]$Name
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Url -Url $Url) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become ready at $Url within $TimeoutSeconds seconds."
}

$startedBackend = $false
$startedFrontend = $false
$backendProcess = $null
$frontendProcess = $null

try {
    if (Test-Url -Url "$BackendUrl/health") {
        Write-Host "Backend is already running at $BackendUrl"
    }
    else {
        $backendProcess = Start-Process `
            -FilePath $PythonPath `
            -ArgumentList @("-u", "-m", "src.assistant.api", "--host", "127.0.0.1", "--port", "$BackendPort") `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $LogRoot "assistant-backend.out.log") `
            -RedirectStandardError (Join-Path $LogRoot "assistant-backend.err.log") `
            -PassThru
        $startedBackend = $true
        Wait-ForUrl -Url "$BackendUrl/health" -TimeoutSeconds 120 -Name "Assistant backend"
    }

    if (Test-Url -Url $FrontendUrl) {
        Write-Host "Frontend is already running at $FrontendUrl"
    }
    else {
        $frontendProcess = Start-Process `
            -FilePath $NpmPath `
            -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
            -WorkingDirectory $FrontendRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $LogRoot "frontend.out.log") `
            -RedirectStandardError (Join-Path $LogRoot "frontend.err.log") `
            -PassThru
        $startedFrontend = $true
        Wait-ForUrl -Url $FrontendUrl -TimeoutSeconds 45 -Name "React frontend"
    }

    [ordered]@{
        started_at = (Get-Date).ToString("o")
        backend_pid = if ($backendProcess) { $backendProcess.Id } else { $null }
        frontend_pid = if ($frontendProcess) { $frontendProcess.Id } else { $null }
        backend_url = $BackendUrl
        frontend_url = $FrontendUrl
        backend_health = "$BackendUrl/health"
    } | ConvertTo-Json | Set-Content -LiteralPath $ProcessFile -Encoding utf8

    Write-Host ""
    Write-Host "Diamond assistant is ready."
    Write-Host "Frontend:       $FrontendUrl/"
    Write-Host "Backend health: $BackendUrl/health"
    Write-Host "Streaming API:  $BackendUrl/chat/stream"
    Write-Host "Logs:           $LogRoot"
}
catch {
    if ($startedFrontend -and $frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force
    }
    if ($startedBackend -and $backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }

    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    foreach ($logName in @("assistant-backend.err.log", "frontend.err.log")) {
        $logPath = Join-Path $LogRoot $logName
        if (Test-Path -LiteralPath $logPath) {
            Write-Host ""
            Write-Host "Last lines from $logPath"
            Get-Content -LiteralPath $logPath -Tail 30
        }
    }
    throw
}
