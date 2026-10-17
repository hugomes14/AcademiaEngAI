param([string]$Python = '3.12')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Instala uv ou cria um venv Python 3.12 manualmente; ver README.md.'
    }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        uv venv --python $Python .venv
        if ($LASTEXITCODE -ne 0) { throw 'Falhou a criação do ambiente.' }
    }
    uv pip install --python '.venv\Scripts\python.exe' --cache-dir '.venv\uv-cache' --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0'
    if ($LASTEXITCODE -ne 0) { throw 'Falhou a instalação do PyTorch CUDA.' }
    uv pip install --python '.venv\Scripts\python.exe' --cache-dir '.venv\uv-cache' --index-url https://pypi.org/simple -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falhou a instalação das dependências.' }
    & '.\.venv\Scripts\python.exe' -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponivel'; x=torch.ones(2,2,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), (x@x).tolist())"
    if ($LASTEXITCODE -ne 0) { throw 'Falhou o teste real de CUDA.' }
} finally {
    Pop-Location
}
