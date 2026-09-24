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
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# --- OTIMIZAÇÕES SIMPLES PARA GPU NVIDIA RTX ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# ----------------------------------------------


def limpar_json(texto: str) -> str:
    """
    Tenta limpar a resposta do modelo para obter apenas JSON válido.

    Remove blocos do tipo:
    ```json
    {...}
    ```

    E também tenta extrair apenas o conteúdo entre o primeiro { e o último }.
    """
    texto = texto.strip()

    texto = re.sub(r"^```json", "", texto, flags=re.IGNORECASE).strip()
    texto = re.sub(r"^```", "", texto).strip()
    texto = re.sub(r"```$", "", texto).strip()

    inicio = texto.find("{")
    fim = texto.rfind("}")

    if inicio != -1 and fim != -1 and fim > inicio:
        texto = texto[inicio:fim + 1]

    return texto.strip()


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


# --- PROMPT PARA CONTAGEM E CORES DOS ANIMAIS ---
prompt = """
Quantos animais existem na imagem e quais são as cores dos animais?

Devolve apenas JSON válido, sem explicações, neste formato exato:

{
  "total_animais": 0,
  "animais": [
    {
      "id": 1,
      "tipo": "gato",
      "cor_principal": "cinzento",
      "cores_secundarias": ["branco"],
      "descricao": "gato cinzento e branco visível na imagem"
    }
  ],
  "descricao_geral": "breve descrição geral dos animais encontrados"
}

Regras:
- Conta apenas animais reais visíveis na imagem.
- Não contes pessoas.
- Não contes objetos, peluches, desenhos, fotografias, pinturas ou estátuas como animais reais.
- Identifica a espécie ou tipo de animal sempre que possível.
- Identifica a cor principal de cada animal.
- Se existirem cores secundárias visíveis, coloca-as em cores_secundarias.
- Se não tiveres a certeza da cor, usa "indefinida".
- Se não tiveres a certeza do tipo de animal, usa "animal desconhecido".
- Se não existirem animais visíveis, devolve total_animais como 0 e animais como lista vazia.
- Não devolvas coordenadas.
- Não devolvas bounding boxes.
- Não desenhes nada na imagem.
"""
# ------------------------------------------------


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


# --- PREPARAÇÃO DOS INPUTS PARA O MODELO ---
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

# Algumas versões/modelos podem devolver token_type_ids, que nem sempre são usados.
inputs.pop("token_type_ids", None)
# ------------------------------------------


# --- EXECUÇÃO ---
print("A analisar os animais e as cores na imagem...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=500,
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
# ---------------


# --- RESPOSTA ORIGINAL DO MODELO ---
print("-" * 50)
print("RESPOSTA ORIGINAL DO MODELO")
print("-" * 50)
print(output_text)
# ----------------------------------


# --- INTERPRETAÇÃO DO JSON ---
json_text = limpar_json(output_text)

try:
    resultado = json.loads(json_text)
except json.JSONDecodeError:
    print("-" * 50)
    print("ERRO: O modelo não devolveu JSON válido.")
    print("Texto recebido após limpeza:")
    print(json_text)
    print("-" * 50)
    raise SystemExit(1)
# -----------------------------


# --- RESULTADO INTERPRETADO ---
print("-" * 50)
print("RESULTADO INTERPRETADO")
print("-" * 50)

total_animais = resultado.get("total_animais", 0)
animais = resultado.get("animais", [])
descricao_geral = resultado.get("descricao_geral", "")

print(f"Total de animais encontrados: {total_animais}")

if animais:
    print("Animais identificados:")

    for animal in animais:
        animal_id = animal.get("id", "?")
        tipo = animal.get("tipo", "desconhecido")
        cor_principal = animal.get("cor_principal", "indefinida")
        cores_secundarias = animal.get("cores_secundarias", [])
        descricao = animal.get("descricao", "")

        print(f"- Animal {animal_id}")
        print(f"  Tipo: {tipo}")
        print(f"  Cor principal: {cor_principal}")

        if cores_secundarias:
            print(f"  Cores secundárias: {', '.join(cores_secundarias)}")
        else:
            print("  Cores secundárias: nenhuma identificada")

        print(f"  Descrição: {descricao}")
else:
    print("Nenhum animal identificado na imagem.")

print(f"Descrição geral: {descricao_geral}")
print("-" * 50)