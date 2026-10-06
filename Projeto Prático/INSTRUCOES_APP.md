# Aplicação RunningAI

## Iniciar

Na raiz do projeto:

```bash
.venv/bin/python app.py
```

Abrir no navegador: <http://127.0.0.1:5000>

Para usar outra porta:

```bash
APP_PORT=8000 .venv/bin/python app.py
```

## Serviços

A página inicial apresenta **RitmoAI** e **PosturaAI** disponíveis. O **TreinadorAI** permanece planeado.

## PosturaAI

Abrir <http://127.0.0.1:5000/postura-ai>, escolher um vídeo e clicar em **Analisar vídeo**. Aceita MP4, MOV, AVI, MKV e WebM até 512 MB, desde que o codec seja suportado pelo OpenCV.

A página acompanha o envio, a fila, os frames processados e a pré-visualização da pose. No fim apresenta o vídeo com os pontos e a classificação, os totais de frames e uma opção de descarga. A classificação é experimental e ainda tem validação fraca. Os resultados não incluem áudio.

O endereço passa a incluir `?job=…`, permitindo voltar à análise ou atualizar a página. Mantém esse endereço para consultar o resultado mais tarde. Vídeos enviados e resultados ficam em `PosturaAI/outputs/web/jobs/`, sem eliminação automática; apagar uma pasta de análise concluída remove os respetivos dados.

O serviço usa estes recursos locais:

- Ambiente `PosturaAI/.venv` preparado conforme a [documentação de treino](PosturaAI/docs/training/training_environment.md).
- Pose: `PosturaAI/outputs/stage1_experimental/best.pth` e o seu `effective_config.py`.
- Classificador: `PosturaAI/outputs/classification/posture_baseline_01/classifier.json`.
- FFmpeg para reprodução no navegador: `PosturaAI/.venv/bin/python -m pip install imageio-ffmpeg==0.6.0` (incluído nas dependências de treino).

Inicia a app normalmente com `.venv/bin/python app.py`. A inferência é executada no ambiente separado do PosturaAI e usa `cuda:0`. Para testar sem GPU, podes iniciar com `POSTURA_DEVICE=cpu .venv/bin/python app.py`; será mais lento.

Executa apenas **um processo** da app: a fila local serializa a inferência na GPU e aceita até quatro análises pendentes. Se a app reiniciar, as análises interrompidas ficam assinaladas para novo envio; os resultados concluídos continuam disponíveis. O log de cada análise fica em `worker.log` na respetiva pasta.

## RitmoAI

- Escolher uma atividade existente em `RitmoAI/Dados/brutos/` ou enviar um ficheiro `.fit`/`.csv`.
- Um CSV deve conter as colunas `distancia_m` e `altitude_m`.
- Selecionar uma zona cardíaca entre Z1 e Z5.
- Consultar o ritmo médio, tempo estimado, distância, subida acumulada, gráfico e previsão por troço.
- Os uploads são processados num ficheiro temporário e eliminados após a previsão.

O modelo e o dataset são recarregados automaticamente quando os respetivos ficheiros são atualizados durante uma nova sessão de treino.
