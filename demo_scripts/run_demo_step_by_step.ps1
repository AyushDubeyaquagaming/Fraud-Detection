$root = "c:/Users/PC/Documents/Fraud Detection"

$steps = [ordered]@{
    "1" = @{ Label = "Rebuild hybrid outputs"; Script = "$root/demo_scripts/01_rebuild_hybrid_outputs.ps1" }
    "2" = @{ Label = "Audit rebuilt outputs"; Script = "$root/demo_scripts/02_audit_hybrid_system.ps1" }
    "3" = @{ Label = "Live score known fraud-linked member"; Script = "$root/demo_scripts/03_live_score_known_member.ps1" }
    "4" = @{ Label = "Live score high-risk unlabeled member"; Script = "$root/demo_scripts/04_live_score_high_risk_unlabeled.ps1" }
    "5" = @{ Label = "Launch Streamlit demo"; Script = "$root/demo_scripts/05_launch_streamlit_demo.ps1" }
}

function Show-Menu {
    Write-Host ""
    Write-Host "Fraud Detection Demo Menu"
    Write-Host "========================="
    foreach ($key in $steps.Keys) {
        Write-Host "$key. $($steps[$key].Label)"
    }
    Write-Host "A. Run steps 1 through 4 in sequence"
    Write-Host "Q. Quit"
    Write-Host ""
}

function Run-Step($choice) {
    $item = $steps[$choice]
    if (-not $item) {
        Write-Host "Invalid selection: $choice"
        return
    }

    Write-Host ""
    Write-Host "Running: $($item.Label)"
    Write-Host "Script : $($item.Script)"
    Write-Host ""
    & $item.Script
}

while ($true) {
    Show-Menu
    $choice = (Read-Host "Select an option").Trim().ToUpper()

    if ($choice -eq "Q") {
        break
    }

    if ($choice -eq "A") {
        foreach ($step in "1", "2", "3", "4") {
            Run-Step $step
            Write-Host ""
            Write-Host "Completed step $step."
            Write-Host ""
        }
        continue
    }

    Run-Step $choice
}