param(
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo
if ((git status --porcelain).Length -gt 0) { throw 'Commit or remove local changes before bundling.' }
$branch = (git branch --show-current).Trim()
$commit = (git rev-parse HEAD).Trim()
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$bundle = Join-Path $OutputDirectory 'worldpolicy-current.bundle'
git bundle create $bundle $branch
if ($LASTEXITCODE -ne 0) { throw 'Bundle creation failed.' }
git bundle verify $bundle
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed.' }
$hash = (Get-FileHash -Algorithm SHA256 $bundle).Hash.ToLowerInvariant()
@{ branch = $branch; commit = $commit; bundle_sha256 = $hash } |
    ConvertTo-Json | Set-Content (Join-Path $OutputDirectory 'bundle_identity.json')
Write-Output "Created verified bundle for $branch at $commit; SHA256 $hash"
