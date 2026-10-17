# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os, math

# Configuração do dispositivo (usa GPU se disponível, senão CPU)
# Nota: O 'map_location' mais à frente tratará da compatibilidade se mudares de PC
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A iniciar inferência no dispositivo: {device}")

# ==============================================================================
# 2. DEFINIÇÃO DA ARQUITETURA (OBRIGATÓRIO REPETIR)
# ==============================================================================
# Precisamos de redefinir as classes exatamente iguais ao treino para carregar os pesos.

_LARGE_NEG = -1e9

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads; self.dk = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        B, Tq, C = q.shape; Tk = k.size(1)
        q = self.q_proj(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None: scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), _LARGE_NEG)
        weights = self.drop(torch.nan_to_num(F.softmax(scores, dim=-1), 0.0))
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.o_proj(ctx)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.h = num_heads; self.dk = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, _LARGE_NEG)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), _LARGE_NEG)
        weights = self.drop(torch.nan_to_num(F.softmax(scores, dim=-1), 0.0))
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(ctx)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_key_padding_mask=None):
        attn = self.self_attention(src, src, src, key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None: attn = attn.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        src = self.norm1(src + self.dropout(attn))
        ff = self.feedforward(src)
        if src_key_padding_mask is not None: ff = ff.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return self.norm2(src + self.dropout(ff))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        sa = self.self_attention(tgt, key_padding_mask=tgt_key_padding_mask)
        if tgt_key_padding_mask is not None: sa = sa.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        tgt = self.norm1(tgt + self.dropout(sa))
        ca = self.cross_attention(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        if tgt_key_padding_mask is not None: ca = ca.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        tgt = self.norm2(tgt + self.dropout(ca))
        ff = self.feedforward(tgt)
        if tgt_key_padding_mask is not None: ff = ff.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        return self.norm3(tgt + self.dropout(ff))

class QATransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_enc=2, num_dec=2,
                 enc_max_len=1024, dec_max_len=256, dim_feedforward=1024, dropout=0.1, tie_weights=True):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_src = nn.Embedding(enc_max_len, d_model)
        self.pos_tgt = nn.Embedding(dec_max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_enc)])
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_dec)])
        self.final_norm_enc = nn.LayerNorm(d_model)
        self.final_norm_dec = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights: self.lm_head.weight = self.tgt_embed.weight

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, S = src_ids.size(); T = tgt_ids.size(1)
        S_cap, T_cap = min(S, self.pos_src.num_embeddings), min(T, self.pos_tgt.num_embeddings)
        pos_s = torch.arange(S_cap, device=src_ids.device).unsqueeze(0).expand(B, S_cap)
        pos_t = torch.arange(T_cap, device=tgt_ids.device).unsqueeze(0).expand(B, T_cap)
        src_ids, tgt_ids = src_ids[:, :S_cap], tgt_ids[:, :T_cap]
        if src_key_padding_mask is not None: src_key_padding_mask = src_key_padding_mask[:, :S_cap]
        if tgt_key_padding_mask is not None: tgt_key_padding_mask = tgt_key_padding_mask[:, :T_cap]
        
        src = self.drop(self.src_embed(src_ids) + self.pos_src(pos_s))
        tgt = self.drop(self.tgt_embed(tgt_ids) + self.pos_tgt(pos_t))
        
        h = src
        for l in self.enc_layers: h = l(h, src_key_padding_mask)
        memory = self.final_norm_enc(h)
        
        y = tgt
        for l in self.dec_layers: y = l(y, memory, tgt_key_padding_mask, src_key_padding_mask)
        y = self.final_norm_dec(y)
        return self.lm_head(y)

# ==============================================================================
# 3. PREPARAÇÃO DO MODELO E TOKENIZER
# ==============================================================================
# Carrega o Tokenizer (tem de ser o mesmo usado no treino)
tok = AutoTokenizer.from_pretrained("t5-small", model_max_length=1024)
PAD = tok.pad_token_id
EOS = tok.eos_token_id
BOS = PAD
SEP = EOS
MAX_SRC_LEN = 1024

# Inicializa a estrutura do modelo (com os mesmos hiperparâmetros do treino!)
model = QATransformer(
    vocab_size=tok.vocab_size,
    d_model=256, nhead=4, num_enc=2, num_dec=2,
    enc_max_len=MAX_SRC_LEN, dec_max_len=64,
    dim_feedforward=1024, dropout=0.0 # Dropout a 0 para inferência
).to(device)

# ==============================================================================
# 4. CARREGAMENTO DOS PESOS (.PT)
# ==============================================================================
# Define o ficheiro que queres carregar (ex: o da última época)
nome_ficheiro = "modelo_squad_epoch_4.pt" 

if os.path.exists(nome_ficheiro):
    print(f"A carregar pesos de {nome_ficheiro}...")
    
    # map_location garante que funciona mesmo se treinou em GPU e agora usas CPU
    state_dict = torch.load(nome_ficheiro, map_location=device)
    model.load_state_dict(state_dict)
    
    # Coloca o modelo em modo de avaliação (desliga Dropout e ajusta Norms)
    model.eval()
    print("Modelo carregado com sucesso!")
else:
    print(f"ERRO: Ficheiro {nome_ficheiro} não encontrado! Corre o script de treino primeiro.")
    exit()

# ==============================================================================
# 5. FUNÇÕES DE SUPORTE (TEXTO -> MODELO -> TEXTO)
# ==============================================================================
def encode_src_inference(question, context):
    """Codifica a pergunta e contexto para tensores."""
    q_ids = tok.encode(question, add_special_tokens=False, truncation=True, max_length=128)
    c_ids = tok.encode(context,  add_special_tokens=False, truncation=True, max_length=890)
    seq = q_ids + [SEP, SEP] + c_ids + [EOS]
    return torch.tensor([seq], dtype=torch.long, device=device)

def decode_output(ids):
    """Converte a saída numérica em texto."""
    return tok.decode([i for i in ids if i != PAD], skip_special_tokens=True)

@torch.no_grad()
def get_answer(model, question, context, max_new_tokens=32):
    """Gera a resposta usando o método Greedy."""
    src = encode_src_inference(question, context)
    tgt = torch.tensor([[BOS]], device=device)
    
    # Loop de geração palavra a palavra
    for _ in range(max_new_tokens):
        # Máscaras
        src_mask = (src == PAD)
        tgt_mask = (tgt == PAD)
        
        logits = model(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        
        # Pega no último token gerado
        next_id = logits[:, -1].argmax(dim=-1)
        
        # Se o modelo "disse" que acabou (EOS), paramos
        if next_id.item() in (EOS, PAD): 
            break
            
        tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)
        
    return decode_output(tgt[0].tolist())

# ==============================================================================
# 6. INTERFACE INTERATIVA (CHAT)
# ==============================================================================
print("\n" + "="*50)
print(" SISTEMA DE PERGUNTAS E RESPOSTAS (SQuAD BOT)")
print("="*50)
print("Escreve 'sair' em qualquer momento para fechar.")

while True:
    print("\n--- NOVO CONTEXTO ---")
    contexto = input("Cola aqui o texto/contexto: ")
    if contexto.lower() in ["sair", "exit"]: break
    
    if len(contexto.strip()) < 5:
        print("Contexto muito curto. Tenta novamente.")
        continue

    while True:
        print("\n--- PERGUNTA (sobre o contexto acima) ---")
        pergunta = input("A tua pergunta: ")
        if pergunta.lower() in ["sair", "exit", "mudar"]: break # 'mudar' pede novo contexto
        
        # O modelo pensa...
        resposta = get_answer(model, pergunta, contexto)
        
        print(f"\n>> RESPOSTA: {resposta}")
        print("-" * 30)