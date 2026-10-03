param([string]$Before = (Join-Path $env:TEMP 'clean_tshc_before_v2.sha256'))
$ErrorActionPreference = 'Stop'
$v2 = Split-Path -Parent $PSScriptRoot
$base = (Resolve-Path -LiteralPath (Join-Path (Split-Path -Parent $v2) 'clean_tshc')).Path
$after = Join-Path $v2 'outputs/audit/clean_tshc_after_v2.sha256'
$rows = Get-ChildItem -LiteralPath $base -Recurse -File -Force | Sort-Object FullName | ForEach-Object { '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.FullName.Substring($base.Length + 1).Replace('\','/') }
[System.IO.File]::WriteAllLines($after, [string[]]$rows, [System.Text.UTF8Encoding]::new($false))
$same = (Get-FileHash -LiteralPath $Before).Hash -eq (Get-FileHash -LiteralPath $after).Hash
Copy-Item -LiteralPath $Before -Destination (Join-Path $v2 'outputs/audit/clean_tshc_before_v2.sha256')
@{ unchanged = $same; files = $rows.Count; before = $Before; after = $after; comparison = 'SHA256 of complete deterministic manifest bytes'; checked_at = (Get-Date).ToUniversalTime().ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $v2 'outputs/audit/protected_verification.json') -Encoding UTF8
if (-not $same) { throw 'CLEAN_TSHC UNCHANGED: NO. Stop and investigate before finishing.' }
Write-Output 'CLEAN_TSHC UNCHANGED: YES'
