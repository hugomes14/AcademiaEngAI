r"""Treino observável de um Transformer causal, por defeito na GPU.

Executar: .venv\Scripts\python.exe train.py --device cuda --steps 800
O corpus incluído é SINTÉTICO e serve para estudar aprendizagem, não fluência.
"""
import argparse
import csv
import itertools
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import MiniTransformer, ModelConfig


class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))
        if len(self.characters) < 2:
            raise ValueError('O corpus precisa de pelo menos dois caracteres distintos.')
        self.to_id = {character: index for index, character in enumerate(self.characters)}

    def encode(self, text):
        try:
            return [self.to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f'Carácter fora do vocabulário: {error.args[0]!r}') from error

    def decode(self, tokens):
        return ''.join(self.characters[int(token)] for token in tokens)


def build_demo_corpus():
    """Frases únicas embaralhadas com semente fixa; sem downloads externos."""
    subjects = ['o modelo', 'a aluna', 'o aluno', 'a professora', 'o professor',
                'a equipa', 'o investigador', 'a investigadora']
    verbs = ['observa', 'estuda', 'analisa', 'compara', 'descreve', 'explora']
    objects = ['o texto', 'a atenção', 'a sequência', 'o contexto', 'o treino',
               'a previsão', 'o resultado', 'a arquitetura', 'o erro', 'a posição']
    endings = ['com cuidado', 'em cada passo', 'durante a aula', 'no laboratório',
               'para aprender', 'com exemplos', 'de forma gradual', 'em detalhe']
    sentences = [f'{subject} {verb} {obj} {ending}.\n'
                 for subject, verb, obj, ending in
                 itertools.product(subjects, verbs, objects, endings)]
    random.Random(42).shuffle(sentences)
    return ''.join(sentences)


def split_corpus(text, context_size):
    boundary = int(len(text) * 0.9)
    newline = text.rfind('\n', 0, boundary)
    if newline >= 0:
        boundary = newline + 1
    train_text, val_text = text[:boundary], text[boundary:]
    if min(len(train_text), len(val_text)) <= context_size:
        raise ValueError('Corpus demasiado curto: treino e validação precisam de '
                         'mais caracteres do que context_size.')
    return train_text, val_text


def make_batch(data, batch_size, context_size, generator):
    if batch_size < 1 or context_size < 1 or len(data) <= context_size:
        raise ValueError('Lote/contexto inválido ou sequência demasiado curta.')
    starts = torch.randint(len(data) - context_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(context_size)[None, :]
    # Os alvos são a mesma sequência deslocada UM carácter para a frente.
    return data[positions], data[positions + 1]


def resolve_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível neste Python. Executa setup_gpu.ps1 e '
                           'usa .venv\\Scripts\\python.exe. CPU só com --device cpu.')
    if name not in ('cuda', 'cpu'):
        raise ValueError('Escolhe --device cuda ou --device cpu.')
    return torch.device(name)


@torch.no_grad()
def evaluate(model, batches):
    was_training = model.training
    model.eval()
    losses, accuracies = [], []
    for x, y in batches:
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
        accuracies.append((logits.argmax(-1) == y).float().mean().item())
    model.train(was_training)
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def run_training(args):
    for name in ('steps', 'eval_every', 'eval_batches', 'batch_size', 'sample_tokens'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} deve ser positivo.')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('lr deve ser positiva e finita.')
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    text = (Path(args.corpus).read_text('utf-8') if args.corpus
            else build_demo_corpus())
    train_text, val_text = split_corpus(text, args.context_size)
    tokenizer = CharacterTokenizer(train_text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    config = ModelConfig(len(tokenizer.characters), args.context_size, args.d_model,
                         args.heads, args.layers, args.dropout)
    model = MiniTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batch_rng = torch.Generator().manual_seed(args.seed)
    evaluation_rng = torch.Generator().manual_seed(args.seed + 1)
    # Lotes FIXOS separados do fluxo de treino: comparar os mesmos exemplos.
    train_batches = [tuple(t.to(device) for t in make_batch(
        train_data, args.batch_size, args.context_size, evaluation_rng))
        for _ in range(args.eval_batches)]
    val_batches = [tuple(t.to(device) for t in make_batch(
        val_data, args.batch_size, args.context_size, evaluation_rng))
        for _ in range(args.eval_batches)]
    prompt = 'o modelo ' if not args.corpus else train_text[:min(20, args.context_size)]
    probe_text = (('o modelo aprende' if not args.corpus else train_text)
                  [:min(24, args.context_size)])
    probe = torch.tensor([tokenizer.encode(probe_text)], device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, samples, attention_snapshots = [], [], []

    def record(step):
        train_loss, train_accuracy = evaluate(model, train_batches)
        val_loss, val_accuracy = evaluate(model, val_batches)
        row = dict(step=step, train_loss=train_loss, val_loss=val_loss,
                   train_accuracy=train_accuracy, val_accuracy=val_accuracy,
                   val_perplexity=math.exp(min(val_loss, 80)),
                   elapsed_seconds=time.perf_counter() - start)
        history.append(row)
        # Mesma semente e prompt em todas as etapas para uma comparação justa.
        sample_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
        generated = model.generate(torch.tensor([tokenizer.encode(prompt)], device=device),
            max_new_tokens=args.sample_tokens, temperature=0.8, generator=sample_rng)
        samples.append(dict(step=step, prompt=prompt,
                            text=tokenizer.decode(generated[0].tolist())))
        model.eval()
        with torch.no_grad():
            _, maps = model(probe, capture_attention=True)
        attention_snapshots.append(torch.stack(maps)[:, 0].cpu().numpy())
        model.train()
        print(f'Passo {step:4d}/{args.steps} | treino {train_loss:.4f} | '
              f'validação {val_loss:.4f} | acerto {val_accuracy:.1%}', flush=True)

    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f'Dispositivo: {device} | {gpu_name} | PyTorch {torch.__version__}', flush=True)
    print(f'Parâmetros: {sum(p.numel() for p in model.parameters()):,} | '
          f'vocabulário: {config.vocab_size} | contexto: {config.context_size}', flush=True)
    record(0)
    model.train()
    for step in range(1, args.steps + 1):
        x, y = (t.to(device) for t in make_batch(
            train_data, args.batch_size, config.context_size, batch_rng))
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        # Há uma previsão por posição: B*T previsões do próximo carácter.
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            record(step)

    model.eval()
    with torch.no_grad():
        probe_logits, _ = model(probe)
    checkpoint = dict(config=asdict(config), characters=tokenizer.characters,
        model={key: value.cpu() for key, value in model.state_dict().items()},
        probe_tokens=probe[0].tolist(), probe_logits=probe_logits.cpu(),
        step=args.steps, seed=args.seed)
    torch.save(checkpoint, output / 'checkpoint.pt')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / 'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), 'utf-8')
    np.savez_compressed(output / 'attention.npz', weights=np.stack(attention_snapshots),
                        steps=np.array([s['step'] for s in samples]),
                        characters=np.array(list(probe_text)))
    # Baseline sem contexto: distribuição de caracteres observada no treino.
    frequencies = torch.bincount(train_data, minlength=config.vocab_size).double()
    unigram_log = (frequencies / frequencies.sum()).clamp_min(1e-12).log()
    unigram_loss = sum((-unigram_log[y.cpu()]).mean().item() for _, y in val_batches)
    unigram_loss /= len(val_batches)
    summary = dict(device_type=device.type, device_name=gpu_name,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        config=asdict(config), steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, parameters=sum(p.numel() for p in model.parameters()),
        train_characters=len(train_data), val_characters=len(val_data),
        corpus='ficheiro do utilizador' if args.corpus else 'sintético: frases combinatórias em português',
        corpus_path=str(Path(args.corpus).resolve()) if args.corpus else None,
        uniform_loss=math.log(config.vocab_size), unigram_val_loss=unigram_loss,
        initial_train_loss=history[0]['train_loss'], initial_val_loss=history[0]['val_loss'],
        final_train_loss=history[-1]['train_loss'], final_val_loss=history[-1]['val_loss'],
        final_val_accuracy=history[-1]['val_accuracy'], final_val_perplexity=history[-1]['val_perplexity'],
        elapsed_seconds=time.perf_counter() - start,
        peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
    from report import create_report
    create_report(output, summary, history, samples, probe_text, np.stack(attention_snapshots))
    print(f'Relatório: {(output / "relatorio.html").resolve()}', flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-batches', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--context-size', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-tokens', type=int, default=360)
    parser.add_argument('--corpus', type=Path, help='Ficheiro UTF-8; omitir usa o corpus sintético.')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs' / 'demo_gpu')
    return parser.parse_args()


if __name__ == '__main__':
    run_training(parse_args())
