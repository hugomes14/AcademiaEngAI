import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_id = "Qwen/Qwen3-VL-2B-Instruct"
image_path = "./images/invoice.png"

# Otimizações para GPU NVIDIA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA não disponível. O modelo será executado em CPU.")

print("A carregar o modelo...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)
print("Pronto. Escreve uma pergunta sobre a imagem ou 'sair' para terminar.")

while True:
    try:
        pergunta = input("\nPergunta: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nA terminar.")
        break

    if pergunta.lower() == "sair":
        break

    if not pergunta:
        print("Escreve uma pergunta.")
        continue

    # Lista nova em cada iteração: não inclui perguntas nem respostas anteriores.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": pergunta},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
        )

    resposta_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    resposta = processor.decode(
        resposta_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"\nResposta:\n{resposta}")