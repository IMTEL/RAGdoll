param(
    [string]$VolumeName = "ragdoll_piper_voices",
    [string[]]$Voices = @("en", "no", "es")
)

$ErrorActionPreference = "Stop"

$voiceFiles = @{
    en = @(
        @{
            Name = "en_US-lessac-medium.onnx"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        },
        @{
            Name = "en_US-lessac-medium.onnx.json"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        }
    )
    no = @(
        @{
            Name = "no_NO-talesyntese-medium.onnx"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx"
        },
        @{
            Name = "no_NO-talesyntese-medium.onnx.json"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx.json"
        }
    )
    es = @(
        @{
            Name = "es_ES-davefx-medium.onnx"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
        },
        @{
            Name = "es_ES-davefx-medium.onnx.json"
            Url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
        }
    )
}

docker volume create $VolumeName | Out-Null

foreach ($voice in $Voices) {
    if (-not $voiceFiles.ContainsKey($voice)) {
        throw "Unknown voice '$voice'. Supported values: $($voiceFiles.Keys -join ', ')"
    }

    foreach ($file in $voiceFiles[$voice]) {
        Write-Host "Downloading $($file.Name) into Docker volume $VolumeName..."
        docker run --rm `
            --user 0:0 `
            -v "${VolumeName}:/voices" `
            curlimages/curl:8.11.1 `
            --fail `
            --location `
            --output "/voices/$($file.Name)" `
            $file.Url

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to download $($file.Name)."
        }
    }
}

Write-Host "Piper voice installation complete."
