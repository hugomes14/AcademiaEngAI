###########################################################
# 1. SETUP, HARDWARE E IMPORTAÇÕES
###########################################################
import sqlite3
from collections import deque
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"SISTEMA: A carregar o motor {model_id} (Aguarde)...")

# Configuração de Hardware
usar_cuda = torch.cuda.is_available()
device_map = "auto" if usar_cuda else None

if usar_cuda and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16 if usar_cuda else torch.float32

print(f"SISTEMA: Dispositivo = {'GPU' if usar_cuda else 'CPU'} | dtype = {dtype}")

# Carregamento do Modelo e Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=dtype,
)

gerador = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Memória de curto prazo (últimas 8 mensagens)
contexto_recente = deque(maxlen=8)

###########################################################
# 2. GESTÃO DE MEMÓRIA DE LONGO PRAZO (SQLITE)
###########################################################
DB_PATH = "viriato_memoria.db"

def inicializar_base_dados():
    """Cria a tabela de memórias se não existir."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS memorias (id INTEGER PRIMARY KEY, facto TEXT NOT NULL)")

def salvar_na_memoria(texto: str):
    """Guarda uma informação na base de dados."""
    texto = (texto or "").strip()
    if not texto: return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO memorias (facto) VALUES (?)", (texto,))

def recuperar_factos_antigos(limite: int = 4) -> str:
    """Recupera as últimas lembranças para injetar no prompt."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT facto FROM memorias ORDER BY id DESC LIMIT ?", (limite,))
        rows = cur.fetchall()
    return "\n".join([f"- {r[0]}" for r in rows]) if rows else "Nenhuma lembrança."

###########################################################
# 3. CAMADA META-COGNITIVA (PENSAMENTO DO CHATBOT)
###########################################################
def analisar_meta_estado(user_input: str):
    msg = (user_input or "").lower()

    if any(p in msg for p in ["romano", "guerra", "ataque", "exército"]):
        estado = "VIGILANTE"
        prompt_sistema = "És o Viriato, guerreiro lusitano. Estás desconfiado. Responde de forma ríspida."
    elif any(p in msg for p in ["vinho", "pão", "amigo", "festa"]):
        estado = "HOSPITALEIRO"
        prompt_sistema = "És o Viriato, taberneiro alegre e generoso. Responde de forma calorosa."
    else:
        estado = "NEUTRO"
        prompt_sistema = "És o Viriato, um taberneiro rústico e equilibrado de Viseu."

    return estado, prompt_sistema

###########################################################
# 4. MOTOR DE GERAÇÃO
###########################################################
def aplicar_chat_template(messages):
    prompt_final = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_final

def gerar_resposta_viriato(pergunta_utilizador: str):
    estado_nome, system_instruction = analisar_meta_estado(pergunta_utilizador)
    memorias_sql = recuperar_factos_antigos()

    system_msg = {
        "role": "system",
        "content": f"{system_instruction}\n\nMemórias: {memorias_sql}\n\nRegras: Fala em pt-pt."
    }

    messages = [system_msg] + list(contexto_recente) + [{"role": "user", "content": pergunta_utilizador}]
    prompt = aplicar_chat_template(messages)

    out = gerador(prompt, max_new_tokens=256, temperature=0.7, do_sample=True, return_full_text=False)
    resposta_ai = out[0].get("generated_text").strip()

    contexto_recente.append({"role": "user", "content": pergunta_utilizador})
    contexto_recente.append({"role": "assistant", "content": resposta_ai})

    return resposta_ai, estado_nome

###########################################################
# 5. LOOP DE INTERAÇÃO
###########################################################
def iniciar_viriato():
    inicializar_base_dados()
    print("\nVIRIATO ATIVADO - Comandos: 'Sair' ou '/memorizar <texto>'\n")

    while True:
        entrada = input("Tu: ").strip()
        if entrada.lower() in ["sair", "exit"]: break

        if entrada.lower().startswith("/memorizar "):
            salvar_na_memoria(entrada[11:])
            print("SISTEMA: Lembrança guardada.")
            continue

        resposta, estado = gerar_resposta_viriato(entrada)
        print(f"\n[ESTADO: {estado}]")
        print(f"Viriato: {resposta}\n")

if __name__ == "__main__":
    iniciar_viriato()
