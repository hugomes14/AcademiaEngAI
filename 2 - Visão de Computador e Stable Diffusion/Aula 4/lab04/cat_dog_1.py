# ==============================================================================
# RESOLUÇÃO DE ERROS DE VERSÃO & ALTERNATIVAS PYTORCH:
# 1) Desinstalar versões antigas em conflito:
#    python -m pip uninstall -y torch torchvision
#
# 2) Reinstalar a combinação correta para Python 3.13.
#    (Podes consultar todas as alternativas em: https://pytorch.org/get-started/locally/)
#
#    OPÇÃO A (Com CUDA 12.6 - Para GPUs NVIDIA recentes):
#    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#
#    OPÇÃO B (Com CUDA 12.4 - Outra alternativa comum no site do PyTorch):
#    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#
#    OPÇÃO C (Apenas CPU - Sem placa gráfica / Não-CUDA):
#    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# 3) Garantir as restantes dependências:
#    python -m pip install --upgrade huggingface-hub accelerate pillow
#
# 4) Para Qwen3-VL, recomenda-se uma versão recente do Transformers:
#    python -m pip uninstall -y transformers
#    python -m pip install git+https://github.com/huggingface/transformers
# ==============================================================================

import json
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# --- OTIMIZAÇÕES SIMPLES PARA GPU NVIDIA RTX ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# ----------------------------------------------


def limpar_json(texto: str) -> str:
    """
    Remove possíveis blocos ```json ... ``` caso o modelo os devolva.
    """
    texto = texto.strip()

    texto = re.sub(r"^```json", "", texto, flags=re.IGNORECASE).strip()
    texto = re.sub(r"^```", "", texto).strip()
    texto = re.sub(r"```$", "", texto).strip()

    return texto


def converter_bbox_1000_para_pixels(box_2d, largura, altura):
    """
    Converte uma bounding box no formato normalizado 0-1000
    para coordenadas reais em píxeis.

    Formato esperado:
    [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = box_2d

    x_min_px = int((x_min / 1000) * largura)
    y_min_px = int((y_min / 1000) * altura)
    x_max_px = int((x_max / 1000) * largura)
    y_max_px = int((y_max / 1000) * altura)

    return [x_min_px, y_min_px, x_max_px, y_max_px]


# --- VERIFICAÇÃO DA GPU ---
print("-" * 50)
print(f"PyTorch detetou CUDA (GPU)? {'SIM' if torch.cuda.is_available() else 'NÃO'}")

if torch.cuda.is_available():
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
else:
    print("O modelo será executado em CPU. Isto será bastante mais lento.")

print("-" * 50)
# --------------------------


model_id = "Qwen/Qwen3-VL-2B-Instruct"
image_path = "./images/test.jpg"
output_image_path = "./images/test_cat_location.jpg"


# --- CARREGAMENTO DO MODELO ---
print("A carregar o modelo Qwen3-VL...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained(model_id)

print(f"O modelo está atualmente alocado no dispositivo: {model.device}")
print("-" * 50)
# -----------------------------


# --- CARREGAMENTO DA IMAGEM ---
image = Image.open(image_path).convert("RGB")
largura, altura = image.size

print(f"Imagem carregada: {largura}x{altura} píxeis")
print("-" * 50)
# ------------------------------


# --- PROMPT PARA LOCALIZAÇÃO DO GATO ---
prompt = """
Localiza o gato na imagem.

Devolve apenas JSON válido, sem explicações, neste formato exato:

{
  "found": true,
  "label": "cat",
  "box_2d": [x_min, y_min, x_max, y_max],
  "center_2d": [x_center, y_center],
  "description": "breve descrição da localização do gato"
}

As coordenadas devem estar na escala normalizada 0-1000.

Se não encontrares nenhum gato, devolve:

{
  "found": false,
  "label": "cat",
  "box_2d": null,
  "center_2d": null,
  "description": "cat not found"
}
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)
inputs.pop("token_type_ids", None)
# --------------------------------------


# --- EXECUÇÃO ---
print("A localizar o gato na imagem...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False
    )

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("-" * 50)
print("RESPOSTA ORIGINAL DO MODELO")
print("-" * 50)
print(output_text)


# --- INTERPRETAÇÃO DO JSON ---
json_text = limpar_json(output_text)

try:
    resultado = json.loads(json_text)
except json.JSONDecodeError:
    print("-" * 50)
    print("ERRO: O modelo não devolveu JSON válido.")
    print("Texto recebido:")
    print(json_text)
    raise SystemExit(1)


print("-" * 50)
print("RESULTADO INTERPRETADO")
print("-" * 50)

if not resultado.get("found"):
    print("Gato não encontrado.")
    raise SystemExit(0)


box_2d = resultado.get("box_2d")
center_2d = resultado.get("center_2d")

if box_2d is None:
    print("Gato indicado como encontrado, mas sem box_2d.")
    raise SystemExit(1)


box_pixels = converter_bbox_1000_para_pixels(box_2d, largura, altura)

x_min, y_min, x_max, y_max = box_pixels
x_center_px = int((x_min + x_max) / 2)
y_center_px = int((y_min + y_max) / 2)

print(f"Label: {resultado.get('label')}")
print(f"Descrição: {resultado.get('description')}")
print(f"Bounding box normalizada 0-1000: {box_2d}")
print(f"Bounding box em píxeis: {box_pixels}")
print(f"Centro em píxeis: [{x_center_px}, {y_center_px}]")


# --- DESENHAR A LOCALIZAÇÃO DO GATO NA IMAGEM ---
annotated_image = image.copy()
draw = ImageDraw.Draw(annotated_image)

# Caixa à volta do gato
draw.rectangle(
    box_pixels,
    outline="red",
    width=4
)

# Ponto central
raio = 6
draw.ellipse(
    [
        x_center_px - raio,
        y_center_px - raio,
        x_center_px + raio,
        y_center_px + raio
    ],
    fill="red"
)

# Texto
draw.text(
    (x_min, max(0, y_min - 20)),
    "cat",
    fill="red"
)

annotated_image.save(output_image_path)

print("-" * 50)
print(f"Imagem com localização guardada em: {output_image_path}")
print("-" * 50)