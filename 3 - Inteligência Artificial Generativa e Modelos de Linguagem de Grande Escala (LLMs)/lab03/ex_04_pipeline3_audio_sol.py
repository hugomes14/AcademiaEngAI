from transformers import pipeline

transcriber = pipeline(model="openai/whisper-base")
result = transcriber("./ficheiros/1.flac")

print(result)