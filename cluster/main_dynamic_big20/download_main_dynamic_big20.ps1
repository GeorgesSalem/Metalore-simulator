$Destination = "C:\Users\LENOVO\Desktop\metalore_main_dynamic_results"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

# Download the complete output directory (models, CSV/JSON files, logs and plots).
scp -r salem@cluster.lip6.fr:~/MetaLore-simulator/main_dynamic_ppo_big20 $Destination

# Download only the newest portable archive for this experiment.
$LatestArchive = ssh salem@cluster.lip6.fr "ls -t ~/MetaLore-simulator/results_backups/main_dynamic_ppo_big20_*.tar.gz 2>/dev/null | head -1"
if ($LatestArchive) {
    scp "salem@cluster.lip6.fr:$LatestArchive" $Destination
}

Write-Host "Downloaded to $Destination"
