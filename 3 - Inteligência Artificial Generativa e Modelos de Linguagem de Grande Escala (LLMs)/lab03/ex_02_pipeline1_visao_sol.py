from transformers import pipeline

vision_classifier = pipeline(task="image-classification")
result = vision_classifier("./ficheiros/prado.jpg")

print(result)