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
#    python -m pip install --upgrade "transformers>=4.45.0" huggingface-hub accelerate pillow
#
# 4) Caso o Qwen3-VL ainda não seja reconhecido pela tua versão do Transformers:
#    python -m pip uninstall -y transformers
#    python -m pip install git+https://github.com/huggingface/transformers
# ==============================================================================

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# --- OTIMIZAÇÕES SIMPLES PARA GPU NVIDIA RTX ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# ----------------------------------------------


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
image_path = "./images/invoice.png"


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


# --- PREPARAÇÃO DA IMAGEM E DO PROMPT ---
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path
            },
            {
                "type": "text",
                "text": (
                    "Faz OCR desta fatura. "
                    "Transcreve todo o texto visível, preservando linhas, valores, datas e identificadores. "
                    "Não inventes informação. "
                    "Se algum campo estiver ilegível, escreve [ilegível]."
                )
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

# Algumas versões/modelos podem devolver token_type_ids, que nem sempre são usados.
inputs.pop("token_type_ids", None)
# ---------------------------------------


# --- EXECUÇÃO DO OCR ---
print("A processar a imagem na GPU..." if torch.cuda.is_available() else "A processar a imagem em CPU...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1200,
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
# -----------------------


# --- RESULTADO ---
print("-" * 50)
print("RESULTADO DO OCR")
print("-" * 50)
print(output_text)