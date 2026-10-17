"""Gráficos exportáveis e relatório HTML autónomo das medições reais."""
import base64
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_report(output, summary, history, samples, probe_text, weights):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#faf9f5', 'axes.facecolor': '#faf9f5'})
    steps = [row['step'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for field, label, color in [('train_loss', 'Treino (avaliação fixa)', '#176b78'),
                                 ('val_loss', 'Validação', '#d5622d')]:
        axes[0].plot(steps, [r[field] for r in history], 'o-', label=label, color=color)
    axes[0].axhline(summary['unigram_val_loss'], color='#888', linestyle='--',
                    label='Baseline sem contexto')
    axes[0].set(xlabel='Atualizações dos pesos', ylabel='Cross-entropy (nats / carácter)',
                title='O erro diminui?')
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [100*r['train_accuracy'] for r in history], 'o-',
                 label='Treino', color='#176b78')
    axes[1].plot(steps, [100*r['val_accuracy'] for r in history], 'o-',
                 label='Validação', color='#d5622d')
    axes[1].set(xlabel='Atualizações dos pesos', ylabel='Próximo carácter correto (%)',
                ylim=(0, 100), title='Quantas previsões acerta?')
    axes[1].legend()
    fig.savefig(output / 'loss.png', dpi=150)
    plt.close(fig)

    chars = ['␣' if c == ' ' else '↵' if c == '\n' else c for c in probe_text]
    heads = weights.shape[2]
    fig, axes = plt.subplots(2, heads, figsize=(max(8, heads*3.4), 7),
                              squeeze=False, constrained_layout=True)
    for row, snapshot in enumerate((0, len(weights)-1)):
        for head in range(heads):
            ax = axes[row, head]
            ax.imshow(weights[snapshot, -1, head], vmin=0, vmax=1, cmap='YlGnBu')
            ax.set_xticks(range(len(chars)), chars, fontsize=8)
            ax.set_yticks(range(len(chars)), chars, fontsize=8)
            ax.set_title(f'Passo {steps[snapshot]} · cabeça {head+1}', fontsize=10)
            ax.set_xlabel('De onde lê (key)')
            if head == 0:
                ax.set_ylabel('Quem lê (query)')
    fig.suptitle('Atenção da última camada · branco acima da diagonal = futuro proibido')
    fig.savefig(output / 'attention.png', dpi=150)
    plt.close(fig)

    config = summary['config']
    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    boxes = [(0.1, 'Caracteres → IDs', 'B × T'),
             (2.5, 'Embedding + posição', f'B × T × {config["d_model"]}'),
             (4.9, f'{config["n_layers"]} blocos Transformer',
              f'LayerNorm → atenção ({config["n_heads"]} cabeças)\n+ residual → LayerNorm → MLP\n+ residual'),
             (7.3, 'LayerNorm + linear', f'B × T × {config["vocab_size"]} logits'),
             (9.7, 'Próximo carácter', 'softmax → amostragem')]
    for index, (x, title, subtitle) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 1.2), 2.1, 1.7, boxstyle='round,pad=0.08',
                                    facecolor='#e8eeeb', edgecolor='#176b78'))
        ax.text(x+1.05, 2.45, title, ha='center', va='center', fontsize=9, weight='bold')
        ax.text(x+1.05, 1.95, subtitle, ha='center', va='center', fontsize=7.5)
        if index < 4:
            ax.annotate('', xy=(x+2.36, 2.1), xytext=(x+2.14, 2.1),
                         arrowprops=dict(arrowstyle='->', color='#176b78'))
    ax.text(6, .5, 'B = tamanho do lote de sequências. T = caracteres por sequência.',
             ha='center', fontsize=10, color='#43535b')
    ax.set_title('O percurso de um lote pelo modelo', fontsize=16, loc='left')
    fig.savefig(output / 'architecture.png', dpi=160)
    plt.close(fig)

    def embedded_image(filename, description):
        encoded = base64.b64encode((output / filename).read_bytes()).decode('ascii')
        return f'<img alt="{html.escape(description)}" src="data:image/png;base64,{encoded}">'

    payload = json.dumps(dict(weights=weights.tolist(), steps=steps,
                              characters=chars, samples=samples), ensure_ascii=True)
    payload = payload.replace('<', '\\u003c')
    escape = html.escape
    document = '''<!doctype html>
<html lang="pt"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Um Transformer por dentro · laboratório</title>
<style>
:root{color-scheme:light;--ink:#172d38;--teal:#176b78;--orange:#bc5024;--paper:#faf9f5}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:56px 28px 80px}header{border-top:5px solid var(--teal);padding-top:24px}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:750}
h1{font-size:clamp(36px,5vw,62px);line-height:1.06;letter-spacing:-.045em;max-width:830px;margin:16px 0 22px}
h2{font-size:29px;line-height:1.2;margin:0 0 16px;letter-spacing:-.025em}h3{font-size:19px;margin:0 0 8px}
p{max-width:880px}section{padding-top:40px;margin-top:36px;border-top:1px solid #d3dcd8}
.lead{font-size:21px;max-width:850px}.muted,small{color:#56656b}a{color:var(--teal)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:32px 0}.stat,.note,.card{background:#eef1eb;padding:20px;border-radius:10px}
.stat strong{display:block;font-size:27px;line-height:1.4}.stat span{font-size:13px;color:#56656b}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{border-left:4px solid var(--orange);background:#f6ece2}
img{display:block;width:100%;height:auto;margin:22px 0;border-radius:8px}code,pre{font:14px/1.65 ui-monospace,Consolas,monospace}
code{background:#e7ece7;padding:2px 5px;border-radius:3px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#172d38;color:#eef5ed;padding:20px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #d3dcd8}
.controls{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0}label{font-size:14px;font-weight:650}select{display:block;padding:8px;font:inherit;border:1px solid #8ca7a8;border-radius:5px;background:white;color:var(--ink)}
.heatmap{overflow:auto;max-width:100%;background:white;border:1px solid #d3dcd8;border-radius:8px}svg{display:block;max-width:100%;height:auto;min-width:430px}
.formula{font:18px/1.8 ui-monospace,monospace;background:#e7ece7;padding:20px;border-radius:8px;overflow:auto}.sample{min-height:135px}
footer{font-size:13px;margin-top:50px;color:#56656b}@media(max-width:700px){main{padding:28px 16px}.stats,.grid{grid-template-columns:1fr 1fr}.stat{padding:14px}.stat strong{font-size:22px}.grid{grid-template-columns:1fr}h2{font-size:25px}}
</style><main><header><div class="eyebrow">Laboratório de modelos de linguagem / 01</div>
<h1>Um Transformer,<br>por dentro.</h1>
<p class="lead">Uma máquina que aprende a completar sequências: vemos as peças, seguimos o treino e observamos para onde a atenção olha.</p>
<p class="muted">Medições reais · __GPU__ · __TORCH__ · semente __SEED__</p></header>
<div class="stats"><div class="stat"><strong>__PARAMS__</strong><span>parâmetros aprendidos</span></div>
<div class="stat"><strong>__STEPS__</strong><span>atualizações dos pesos</span></div>
<div class="stat"><strong>__VAL__</strong><span>perda final de validação</span></div>
<div class="stat"><strong>__TIME__ s</strong><span>treino + avaliações + exportação de dados</span></div></div>
<div class="note"><strong>Como interpretar esta experiência.</strong> Corpus: __CORPUS__. A demonstração incluída combina frases curtas e repetitivas. Aprender essas regularidades não equivale a compreender português nem permite esperar a capacidade de um LLM. O treino usa __TRAINCHARS__ caracteres; a validação, __VALCHARS__ caracteres de uma parte separada antes de construir janelas.</div>
<section><div class="eyebrow">01 / Arquitetura</div><h2>Uma mesa de trabalho com várias pistas.</h2>
<p>Imagina completar uma frase de olhos postos apenas no que já foi escrito. Cada cabeça de atenção é um leitor que pode procurar pistas diferentes. O modelo junta as pistas e atribui uma pontuação a cada possível próximo carácter.</p>
__ARCHITECTURE__
<div class="grid"><div class="card"><h3>Representar e localizar</h3><p>Cada carácter recebe um ID. O embedding converte-o num vetor aprendido; somamos um vetor de posição para distinguir onde aparece na janela.</p></div>
<div class="card"><h3>Relacionar e transformar</h3><p>Cada bloco aplica atenção e depois uma MLP, que transforma cada posição separadamente. LayerNorm estabiliza as escalas; as somas residuais preservam um caminho direto para informação e gradientes.</p></div></div>
<p>Este modelo tem <strong>__LAYERS__ camadas, __HEADS__ cabeças por camada, embeddings de __DMODEL__ valores e contexto de __CONTEXT__ caracteres</strong>. As posições são aprendidas. É um decoder causal; não existe encoder nem atenção cruzada. O código usa operações matriciais explícitas, sem atenção pronta ou pesos pré-treinados.</p>
</section><section><div class="eyebrow">02 / Atenção</div><h2>Que informação é útil para esta posição?</h2>
<p><strong>Query (Q)</strong> é o que uma posição procura. <strong>Key (K)</strong> é a pista que cada posição oferece. <strong>Value (V)</strong> é o conteúdo que pode ser trazido para a mistura. São três projeções lineares dos mesmos vetores de entrada.</p>
<div class="formula">Q = XWq &nbsp; K = XWk &nbsp; V = XWv<br>A = softmax(QKᵀ / √d_cabeça + máscara)<br>contexto = A V</div>
<p>O produto QKᵀ mede compatibilidade; a divisão controla a escala. A máscara coloca −∞ no futuro, que vira probabilidade zero após softmax. Cada linha de A soma 1. A diagonal é permitida: usamos o carácter atual para prever o seguinte.</p>
<div class="controls"><label>Etapa<select id="snapshot"></select></label><label>Camada<select id="layer"></select></label><label>Cabeça<select id="head"></select></label></div>
<div class="heatmap"><svg id="map" role="img" aria-label="Matriz de atenção; linhas são queries e colunas são keys"></svg></div>
<p id="cell" class="muted">Passa o rato ou usa Tab sobre uma célula para ler o peso. ␣ representa um espaço.</p>
<p class="note">Atenção não é uma explicação causal completa da previsão. Os pesos mostram uma mistura de informação numa cabeça; os valores, a projeção, a MLP e as restantes camadas também influenciam a saída. As cabeças não recebem papéis linguísticos definidos à partida.</p>
</section><section><div class="eyebrow">03 / Treino</div><h2>Prever, medir o erro, ajustar, repetir.</h2>
<pre>Texto:       o modelo\nEntrada x:   o model\nAlvo y:       modelo\n             ↑ cada posição prevê o carácter seguinte</pre>
<p>Cada lote contém __BATCH__ janelas de __CONTEXT__ caracteres. Para todas as posições, calculamos cross-entropy entre os logits e o próximo carácter real. Não aplicamos softmax antes desta perda: o PyTorch já o incorpora de forma numericamente estável.</p>
<pre>optimizer.zero_grad()   # limpar gradientes anteriores\nlogits, _ = model(x)    # previsões para cada posição\nloss = cross_entropy(logits, y)\nloss.backward()         # derivadas em relação aos pesos\nclip_grad_norm_(...)    # limitar a norma global a 1\noptimizer.step()        # AdamW atualiza os pesos</pre>
<p>AdamW usa taxa de aprendizagem __LR__. O dropout está ativo no treino; na avaliação é desligado. Os pesos começam aleatórios e permanecem fixos durante geração. A mesma semente ajuda a reproduzir o ensaio, sem garantir identidade bit a bit entre placas ou versões.</p>
</section><section><div class="eyebrow">04 / Evolução</div><h2>Melhora no treino e em exemplos reservados?</h2>
__LOSS__
<p>As curvas usam sempre os mesmos lotes de avaliação, sem dropout. Não comparam diretamente um lote aleatório de treino com outro diferente de validação. A baseline ignora o contexto e usa apenas frequências dos caracteres do treino.</p>
<table><thead><tr><th>Medição</th><th>Inicial</th><th>Final</th></tr></thead><tbody>
<tr><td>Perda de treino</td><td>__INITIALTRAIN__</td><td>__FINALTRAIN__</td></tr>
<tr><td>Perda de validação</td><td>__INITIALVAL__</td><td>__VAL__</td></tr>
<tr><td>Acerto na validação</td><td>__INITIALACC__</td><td>__ACC__</td></tr></tbody></table>
<p>Perplexidade final: <strong>__PPL__</strong> (= exp da perda). Valores menores são melhores neste mesmo corpus e vocabulário; não são comparáveis diretamente com modelos que usam outra tokenização. Se o erro de treino descer enquanto o de validação sobe, podemos estar a observar sobreajuste.</p>
<h3>O mesmo início, em diferentes momentos</h3><label>Etapa da geração<select id="sample"></select></label><pre id="sampletext" class="sample"></pre>
<p class="muted">Mesmo prompt, temperatura 0,8 e semente de amostragem. A temperatura altera a distribuição das escolhas; não volta a treinar o modelo.</p>
</section><section><div class="eyebrow">05 / Experimentar</div><h2>Muda uma peça de cada vez.</h2>
<p>1. Compara os passos 0 e final na mesma cabeça: o que mudou?<br>2. Compara treino e validação: existe distância entre as curvas?<br>3. Reduz o contexto e repete: que pistas deixam de estar acessíveis?<br>4. Experimenta outro corpus UTF-8 e compara com um novo treino, sem confundir tarefas diferentes.</p>
<p>Memória máxima de tensores CUDA alocados: __VRAM__ MB. Este valor não é o consumo total mostrado no Gestor de Tarefas. O checkpoint inclui pesos, configuração e vocabulário para voltar a gerar texto; não inclui o estado do otimizador para retomar exatamente o treino.</p>
<p>Lê <a href="../../model.py">model.py</a> para a arquitetura e <a href="../../train.py">train.py</a> para o treino. Os gráficos também estão guardados como PNG, as métricas como CSV e as matrizes em attention.npz.</p>
</section><footer>Relatório local, autónomo, sem pedidos à rede. Os resultados descrevem esta execução, não uma promessa de generalização.</footer>
<script id="experiment" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('experiment').textContent);
const $=id=>document.getElementById(id);
function choices(id,values){for(const [i,label] of values.entries()){
const o=document.createElement('option');o.value=i;o.textContent=label;$(id).appendChild(o)}}
choices('snapshot',data.steps.map(s=>'Passo '+s));
choices('layer',data.weights[0].map((_,i)=>'Camada '+(i+1)));
choices('head',data.weights[0][0].map((_,i)=>'Cabeça '+(i+1)));
choices('sample',data.steps.map(s=>'Passo '+s));
$('snapshot').value=data.steps.length-1;$('sample').value=data.steps.length-1;
function cellMessage(r,c,w){$('cell').textContent='Posição '+(r+1)+' ('+data.characters[r]+') lê posição '+(c+1)+' ('+data.characters[c]+'): '+(100*w).toFixed(2)+'%'+(c>r?' · futuro mascarado':'')}
function draw(){const matrix=data.weights[+$('snapshot').value][+$('layer').value][+$('head').value],n=matrix.length;
const svg=$('map'),ns='http://www.w3.org/2000/svg',cell=30,margin=65;svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${margin+n*cell+15} ${margin+n*cell+20}`);
function text(x,y,s){const el=document.createElementNS(ns,'text');el.setAttribute('x',x);el.setAttribute('y',y);el.setAttribute('font-size','13');el.setAttribute('text-anchor','middle');el.textContent=s;svg.appendChild(el)}
for(let r=0;r<n;r++){text(margin-22,margin+r*cell+20,data.characters[r]);text(margin+r*cell+15,margin-18,data.characters[r]);
for(let c=0;c<n;c++){let w=matrix[r][c],rect=document.createElementNS(ns,'rect');rect.setAttribute('x',margin+c*cell);rect.setAttribute('y',margin+r*cell);rect.setAttribute('width',cell-1);rect.setAttribute('height',cell-1);rect.setAttribute('fill',c>r?'#f3f5f1':`rgb(${Math.round(234-214*w)},${Math.round(242-135*w)},${Math.round(225-105*w)})`);rect.setAttribute('tabindex','0');rect.setAttribute('aria-label',`Posição ${r+1} lê ${c+1}: ${(100*w).toFixed(2)}%`);rect.addEventListener('mouseenter',()=>cellMessage(r,c,w));rect.addEventListener('focus',()=>cellMessage(r,c,w));const title=document.createElementNS(ns,'title');title.textContent=(100*w).toFixed(2)+'%';rect.appendChild(title);svg.appendChild(rect)}}}
for(const id of ['snapshot','layer','head'])$(id).addEventListener('change',draw);
function sample(){$('sampletext').textContent=data.samples[+$('sample').value].text}
$('sample').addEventListener('change',sample);draw();sample();
</script></main></html>'''
    replacements = {
        '__GPU__': escape(summary['device_name']), '__TORCH__': escape(summary['torch_version']),
        '__SEED__': str(summary['seed']), '__PARAMS__': f"{summary['parameters']:,}",
        '__STEPS__': str(summary['steps']), '__VAL__': f"{summary['final_val_loss']:.3f}",
        '__TIME__': f"{summary['elapsed_seconds']:.1f}", '__CORPUS__': escape(summary['corpus']),
        '__TRAINCHARS__': str(summary['train_characters']), '__VALCHARS__': str(summary['val_characters']),
        '__LAYERS__': str(config['n_layers']), '__HEADS__': str(config['n_heads']),
        '__DMODEL__': str(config['d_model']), '__CONTEXT__': str(config['context_size']),
        '__BATCH__': str(summary['batch_size']), '__LR__': str(summary['learning_rate']),
        '__INITIALTRAIN__': f"{summary['initial_train_loss']:.3f}",
        '__FINALTRAIN__': f"{summary['final_train_loss']:.3f}",
        '__INITIALVAL__': f"{summary['initial_val_loss']:.3f}",
        '__INITIALACC__': f"{100*history[0]['val_accuracy']:.1f}%",
        '__ACC__': f"{100*summary['final_val_accuracy']:.1f}%",
        '__PPL__': f"{summary['final_val_perplexity']:.2f}",
        '__VRAM__': f"{summary['peak_vram_mb']:.1f}", '__PAYLOAD__': payload,
        '__ARCHITECTURE__': embedded_image('architecture.png', 'Arquitetura do Transformer'),
        '__LOSS__': embedded_image('loss.png', 'Curvas de perda e acerto ao longo do treino'),
    }
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    (output / 'relatorio.html').write_text(document, encoding='utf-8')
