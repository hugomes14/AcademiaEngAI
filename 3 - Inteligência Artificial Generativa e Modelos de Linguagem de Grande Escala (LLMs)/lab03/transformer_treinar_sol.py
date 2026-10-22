# ==============================================================================
# 1. IMPORTAÇÕES E CONFIGURAÇÃO
# ==============================================================================
# Importa módulos para interagir com o sistema operativo (os), JSON e matemática
import os, json, math, random

# Importa o PyTorch, a principal biblioteca de Deep Learning usada aqui
import torch
import torch.nn as nn                 # Redes neuronais (camadas, funções de perda)
import torch.nn.functional as F       # Funções sem estado (ativações, softmax)
from torch.utils.data import Dataset, DataLoader # Ferramentas para gerir dados em lotes

# Importa da biblioteca HuggingFace para processar texto (tokenização)
from transformers import AutoTokenizer  

# Importa bibliotecas para cálculo numérico e gráficos
import numpy as np
import matplotlib.pyplot as plt

# Define o dispositivo de cálculo.
# 'cuda' usa a placa gráfica (NVIDIA), que é muito mais rápida para matrizes.
# Se não houver, usa o processador central (cpu).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"A usar o dispositivo: {device}")

# Define sementes (seeds) fixas.
# Isto garante que os números aleatórios (pesos iniciais, baralhamento) 
# sejam os mesmos sempre que corres o código, permitindo reproduzir resultados.
torch.manual_seed(42); random.seed(42)

# ==============================================================================
# 2. CARREGAMENTO DOS DADOS (SQuAD)
# ==============================================================================
# Caminho para o ficheiro JSON do dataset SQuAD (Stanford Question Answering Dataset)
# Caminho relativo à pasta do script (funciona independentemente da pasta de onde é executado)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "conteudo", "squad", "train-v1.1.json")

# Verifica se o ficheiro existe para evitar erros
if not os.path.exists(json_path):
    print(f"AVISO: Ficheiro {json_path} não encontrado. Verifica o caminho.")
else:
    # Abre e carrega o JSON para a memória
    with open(json_path, "r", encoding="utf-8") as f:
        squad_train = json.load(f)

    # Inicializa listas vazias para guardar os dados processados
    questions, answers, contexts = [], [], []
    
    # Navega pela estrutura aninhada do JSON do SQuAD:
    # Artigo -> Parágrafos -> (Contexto + Perguntas/Respostas)
    for art in squad_train["data"]:
        for para in art["paragraphs"]:
            ctx = para.get("context", "") # O texto base (parágrafo)
            for qa in para["qas"]:
                if qa.get("answers"): # Só queremos perguntas com resposta
                    questions.append(qa["question"])
                    # Pega apenas na primeira resposta disponível (texto)
                    answers.append(qa["answers"][0]["text"])
                    contexts.append(ctx)

    print(f"Total de exemplos carregados: {len(questions)}")

# ==============================================================================
# 3. TOKENIZAÇÃO E CODIFICAÇÃO
# ==============================================================================
# Carrega o Tokenizer do modelo T5.
# 'model_max_length=1024' previne avisos de erro se o texto for muito longo.
tok = AutoTokenizer.from_pretrained("t5-small", model_max_length=1024)

# IDs especiais que o modelo usa para controlo:
PAD = tok.pad_token_id      # 0: Padding (preenchimento vazio)
EOS = tok.eos_token_id      # 1: End Of Sequence (fim da frase)
BOS = PAD                   # T5 usa PAD como início para o Decoder
SEP = EOS                   # Usaremos o token de fim como separador

# Definição dos limites de tamanho para entrada (src) e saída (tgt)
MAX_SRC_LEN = 1024
MAX_TGT_LEN = 64

def encode_src(question, context, max_q=64, max_c=896):
    """
    Transforma texto em números (IDs) para a entrada do Encoder.
    Formato: Pergunta + [SEP] + Contexto + [EOS]
    """
    
    # Converte texto em IDs. 'truncation=True' corta se for gigante.
    # 'add_special_tokens=False' porque vamos adicionar os nossos manualmente.
    q_ids = tok.encode(question, add_special_tokens=False, truncation=True, max_length=max_q*4)
    c_ids = tok.encode(context,  add_special_tokens=False, truncation=True, max_length=max_c*4)
    
    # Concatena: [Pergunta] [SEP] [SEP] [Contexto] [EOS]
    seq = q_ids + [SEP, SEP] + c_ids + [EOS]
    
    # Lógica de "Orçamento" (Budget):
    # Se a soma da pergunta + contexto exceder 1024, temos de cortar.
    if len(seq) > MAX_SRC_LEN:
        budget = MAX_SRC_LEN - 3 # Espaço para tokens especiais
        
        # Tentamos manter a pergunta toda, mas limitamo-la a 1/3 do orçamento
        keep_q = min(len(q_ids), min(max_q, budget // 3))
        
        # O resto do espaço vai para o contexto (o contexto é cortado no fim)
        keep_c = budget - keep_q
        
        # Recorta as listas
        q_ids = q_ids[:keep_q]; c_ids = c_ids[:keep_c]
        
        # Reconstrói a sequência final
        seq = q_ids + [SEP, SEP] + c_ids + [EOS]
    return seq

def encode_tgt(answer, max_a=32):
    """
    Prepara a resposta alvo para o Decoder.
    Formato: [BOS] + Resposta + [EOS]
    """
    a_ids = tok.encode(answer, add_special_tokens=False)[:max_a]
    seq = [BOS] + a_ids + [EOS]
    return seq[:MAX_TGT_LEN]

def decode(ids):
    """Função inversa: transforma números em texto legível."""
    # Filtra os tokens de PAD para não aparecerem no texto final
    return tok.decode([i for i in ids if i != PAD], skip_special_tokens=True)

# ==============================================================================
# 4. DATASET E DATALOADER
# ==============================================================================
class QADataset(Dataset):
    """Classe que o PyTorch usa para aceder aos dados um a um."""
    def __init__(self, qs, ans, ctx): self.qs, self.ans, self.ctx = qs, ans, ctx
    def __len__(self): return len(self.qs) # Tamanho total
    def __getitem__(self, i):
        # Quando pedimos o item 'i', ele codifica o texto on-the-fly
        return (
            torch.tensor(encode_src(self.qs[i], self.ctx[i]), dtype=torch.long),
            torch.tensor(encode_tgt(self.ans[i]), dtype=torch.long)
        )

def pad_collate(batch):
    """
    Prepara um lote (batch) de dados.
    Como as frases têm tamanhos diferentes, temos de preencher com PADs
    para criar uma matriz retangular perfeita.
    """
    srcs, tgts = zip(*batch) # Separa perguntas e respostas
    
    # Descobre qual é a frase mais longa neste lote específico
    max_s, max_t = max(len(s) for s in srcs), max(len(t) for t in tgts)
    
    # Cria matrizes cheias de zeros (PAD)
    src_pad = torch.full((len(srcs), max_s), PAD, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_t), PAD, dtype=torch.long)
    
    # Copia os dados reais para cima dos zeros
    for i,(s,t) in enumerate(zip(srcs,tgts)):
        src_pad[i,:len(s)] = s; tgt_pad[i,:len(t)] = t
        
    # Envia os tensores finais para a GPU (ou CPU)
    return src_pad.to(device), tgt_pad.to(device)

# Divide os dados: 90% para treinar, 10% para validar se aprendeu
if 'questions' in locals():
    N = len(questions); split = int(0.9*N)
    train_ds = QADataset(questions[:split], answers[:split], contexts[:split])
    val_ds   = QADataset(questions[split:], answers[split:], contexts[split:])
    
    # DataLoaders gerem o carregamento em paralelo e misturam os dados (shuffle)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=pad_collate)
    val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=pad_collate)

# ==============================================================================
# 5. ARQUITETURA DO MODELO (TRANSFORMER)
# ==============================================================================
_LARGE_NEG = -1e9 # Número muito negativo para usar nas máscaras (Softmax de -infinito = 0)



class MultiHeadAttention(nn.Module):
    """
    O coração do Transformer. Permite que cada palavra 'olhe' para todas as outras
    para perceber o contexto.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads; self.dk = embed_dim // num_heads # Dimensão por cabeça
        
        # Projeções lineares (pesos aprendíveis) para Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False) # Projeção de saída
        self.drop = nn.Dropout(dropout)
        
    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        B, Tq, C = q.shape; Tk = k.size(1)
        
        # Projeta e divide em cabeças (Heads): [Batch, Tempo, Cabeças, Dimensão]
        q = self.q_proj(q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.h, self.dk).transpose(1, 2)
        
        # Fórmula da Atenção: (Q * K^T) / raiz(dimensão)
        # Calcula a similaridade entre Queries e Keys
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        
        # Aplica máscaras (se necessário)
        if attn_mask is not None: scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Máscara de Padding: Garante que o modelo ignora os tokens de PAD (zeros)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), _LARGE_NEG)
            
        # Softmax: Converte scores em probabilidades (soma = 1)
        weights = self.drop(torch.nan_to_num(F.softmax(scores, dim=-1), 0.0))
        
        # Multiplica pelos Values (V) para obter o contexto final
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.o_proj(ctx)

class MaskedMultiHeadAttention(nn.Module):
    """
    Versão da atenção para o Decoder. Usa uma máscara 'causal' para impedir
    que o modelo veja o futuro (palavras seguintes) durante o treino.
    """
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
        
        # Máscara Causal (Triângulo inferior):
        # Permite ver o passado, bloqueia o futuro.
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, _LARGE_NEG)
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), _LARGE_NEG)
            
        weights = self.drop(torch.nan_to_num(F.softmax(scores, dim=-1), 0.0))
        ctx = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(ctx)

class TransformerEncoderLayer(nn.Module):
    """Uma camada do Encoder. Processa a entrada para criar uma 'memória'."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        # Rede FeedForward simples para processar a informação
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        # Normalização por camada (estabiliza o treino)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_key_padding_mask=None):
        # 1. Atenção + Resíduo + Norm
        attn = self.self_attention(src, src, src, key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None: attn = attn.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        src = self.norm1(src + self.dropout(attn))
        
        # 2. FeedForward + Resíduo + Norm
        ff = self.feedforward(src)
        if src_key_padding_mask is not None: ff = ff.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return self.norm2(src + self.dropout(ff))

class TransformerDecoderLayer(nn.Module):
    """Uma camada do Decoder. Gera texto usando a memória do Encoder."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. Self-Attention (Masked): Olha para o texto que já gerou
        sa = self.self_attention(tgt, key_padding_mask=tgt_key_padding_mask)
        if tgt_key_padding_mask is not None: sa = sa.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        tgt = self.norm1(tgt + self.dropout(sa))
        
        # 2. Cross-Attention: Olha para a memória do Encoder (Q=Decoder, K,V=Encoder)
        ca = self.cross_attention(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        if tgt_key_padding_mask is not None: ca = ca.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        tgt = self.norm2(tgt + self.dropout(ca))
        
        # 3. FeedForward
        ff = self.feedforward(tgt)
        if tgt_key_padding_mask is not None: ff = ff.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        return self.norm3(tgt + self.dropout(ff))

class QATransformer(nn.Module):
    """O modelo completo que junta tudo."""
    def __init__(self, vocab_size, d_model=256, nhead=4, num_enc=2, num_dec=2,
                 enc_max_len=1024, dec_max_len=256, dim_feedforward=1024, dropout=0.1, tie_weights=True):
        super().__init__()
        # Embeddings: Transformam IDs (números inteiros) em vetores densos
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        
        # Embeddings Posicionais: Ensinam ao modelo a ordem das palavras
        self.pos_src = nn.Embedding(enc_max_len, d_model)
        self.pos_tgt = nn.Embedding(dec_max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Cria listas de camadas (Stack)
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_enc)])
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_dec)])
        
        self.final_norm_enc = nn.LayerNorm(d_model)
        self.final_norm_dec = nn.LayerNorm(d_model)
        
        # Cabeça final: Projeta o vetor de volta para o tamanho do vocabulário
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: Usa os mesmos pesos na entrada e saída (poupa memória)
        if tie_weights: self.lm_head.weight = self.tgt_embed.weight
        
        # Inicialização Xavier (ajuda o treino a começar bem)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, S = src_ids.size(); T = tgt_ids.size(1)
        
        # Cria índices de posição (0, 1, 2, ...)
        S_cap, T_cap = min(S, self.pos_src.num_embeddings), min(T, self.pos_tgt.num_embeddings)
        pos_s = torch.arange(S_cap, device=src_ids.device).unsqueeze(0).expand(B, S_cap)
        pos_t = torch.arange(T_cap, device=tgt_ids.device).unsqueeze(0).expand(B, T_cap)
        
        # Trunca para garantir compatibilidade
        src_ids, tgt_ids = src_ids[:, :S_cap], tgt_ids[:, :T_cap]
        if src_key_padding_mask is not None: src_key_padding_mask = src_key_padding_mask[:, :S_cap]
        if tgt_key_padding_mask is not None: tgt_key_padding_mask = tgt_key_padding_mask[:, :T_cap]
        
        # Soma Embedding da palavra + Embedding da posição
        src = self.drop(self.src_embed(src_ids) + self.pos_src(pos_s))
        tgt = self.drop(self.tgt_embed(tgt_ids) + self.pos_tgt(pos_t))
        
        # Passa pelo Encoder
        h = src
        for l in self.enc_layers: h = l(h, src_key_padding_mask)
        memory = self.final_norm_enc(h) # O output do Encoder é a 'memória'
        
        # Passa pelo Decoder (usando a memória)
        y = tgt
        for l in self.dec_layers: y = l(y, memory, tgt_key_padding_mask, src_key_padding_mask)
        y = self.final_norm_dec(y)
        
        # Gera probabilidades para a próxima palavra
        return self.lm_head(y)

# ==============================================================================
# 6. INSTANCIAÇÃO DO MODELO
# ==============================================================================
# Cria o objeto do modelo com os hiperparâmetros definidos
model = QATransformer(
    vocab_size=tok.vocab_size,
    d_model=256, nhead=4, num_enc=2, num_dec=2,
    enc_max_len=MAX_SRC_LEN, dec_max_len=MAX_TGT_LEN,
    dim_feedforward=1024, dropout=0.1
).to(device) # Move o modelo para a GPU

print(f"Modelo criado com {sum(p.numel() for p in model.parameters()):,} parâmetros.")

# ==============================================================================
# 7. FUNÇÕES AUXILIARES DE TREINO E INFERÊNCIA
# ==============================================================================
# Otimizador AdamW: O algoritmo que atualiza os pesos para minimizar o erro
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

def _step_batch(model, batch):
    """Executa um passo de treino."""
    src, tgt_full = batch
    
    # Teacher Forcing:
    # Input do Decoder: Tudo menos o último token (começa com BOS)
    # Target (Alvo): Tudo menos o primeiro token (o que queremos prever)
    tgt_in, tgt_out = tgt_full[:, :-1], tgt_full[:, 1:] 
    
    # Máscaras de Padding
    src_mask = (src == PAD); tgt_mask = (tgt_in == PAD)
    
    # Forward Pass (Previsão)
    logits = model(src, tgt_in, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
    
    # Calcula o erro (Loss) comparando previsão com a realidade
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)

@torch.no_grad()
def evaluate(model, loader):
    """Calcula o erro médio no conjunto de validação sem atualizar pesos."""
    model.eval(); tot = 0.0; cnt = 0
    for batch in loader:
        tot += _step_batch(model, batch).item()
        cnt += 1
    model.train() # Volta ao modo de treino
    return tot / max(1, cnt)

@torch.no_grad()
def greedy_answer(model, question, context, max_new=32):
    """
    Geração de texto para inferência.
    Gera um token de cada vez, alimentando-o de volta ao modelo.
    """
    model.eval()
    src = torch.tensor([encode_src(question, context)], device=device)
    tgt = torch.tensor([[BOS]], device=device) # Começa com BOS
    
    for _ in range(max_new):
        # Prevê o próximo token
        logits = model(src, tgt, src_key_padding_mask=(src==PAD), tgt_key_padding_mask=(tgt==PAD))
        next_id = logits[:, -1].argmax(dim=-1) # Escolhe o mais provável
        
        # Para se encontrar EOS ou PAD
        if next_id.item() in (EOS, PAD): break
        
        # Adiciona à sequência
        tgt = torch.cat([tgt, next_id.unsqueeze(1)], dim=1)
        
    model.train()
    return decode(tgt[0].tolist())

# Scheduler (Gestão da Taxa de Aprendizagem)
# Começa devagar (warmup) e depois diminui (cosine decay) para estabilidade
EPOCHS = 4
total_train_steps = EPOCHS * len(train_dl)
warmup_steps = max(100, int(0.06 * total_train_steps))
def lr_lambda(step):
    if step < warmup_steps: return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
    return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * progress)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

# ==============================================================================
# 8. LOOP DE TREINO (MAIN LOOP)
# ==============================================================================
loss_history = []
print("\n--- A INICIAR TREINO ---")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for i, batch in enumerate(train_dl, 1):
        optim.zero_grad()               # Limpa gradientes antigos
        loss = _step_batch(model, batch)# Calcula perda
        loss.backward()                 # Backpropagation (calcula gradientes)
        
        # Clip Gradient: Evita explosão de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optim.step(); scheduler.step()  # Atualiza pesos e Learning Rate
        
        loss_history.append(loss.item())
        running += loss.item()

    # Validação no final da época
    val_loss = evaluate(model, val_dl)
    print(f"Epoch {epoch:02d} | Train Loss {running/i:.4f} | Val Loss {val_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")
    
    # Guarda o estado do modelo
    torch.save(model.state_dict(), os.path.join(BASE_DIR, f"modelo_squad_epoch_{epoch}.pt"))
    print(f"-> Modelo guardado: modelo_squad_epoch_{epoch}.pt")

# ==============================================================================
# 9. TESTE E INFERÊNCIA (PÓS-TREINO)
# ==============================================================================
print("\n--- A TESTAR MODELO TREINADO ---")
for i in range(5):
    q, c, a = val_ds.qs[i], val_ds.ctx[i], val_ds.ans[i]
    pred = greedy_answer(model, q, c)
    print(f"\nQ:  {q}\nReal: {a}\nPrev: {pred}")

# ==============================================================================
# 10. VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================
# Função para suavizar o gráfico (média móvel)
def moving_average(x, k=200):
    if len(x) < k: return x
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[k:] - c[:-k]) / float(k)

plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Perda (Raw)", alpha=0.3)
ma = moving_average(loss_history, k=200)
if len(ma) > 0:
    plt.plot(range(len(loss_history) - len(ma), len(loss_history)), ma, label="Média Móvel", color='red')
plt.title("Evolução do Treino")
plt.xlabel("Passos")
plt.ylabel("Loss")
plt.legend(); plt.grid(True)
plt.show()