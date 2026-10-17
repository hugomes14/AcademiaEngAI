"""Gerar texto a partir de um checkpoint local, sem voltar a treinar."""
import argparse
from pathlib import Path

import torch

from model import MiniTransformer, ModelConfig
from train import CharacterTokenizer, resolve_device


def generate_from_checkpoint(path, prompt, max_new_tokens=200, temperature=0.8,
                             device_name='cuda', seed=42):
    device = resolve_device(device_name)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = MiniTransformer(ModelConfig(**checkpoint['config'])).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    tokenizer = CharacterTokenizer(''.join(checkpoint['characters']))
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = model.generate(tokens, max_new_tokens=max_new_tokens,
                             temperature=temperature, generator=generator)
    return tokenizer.decode(result[0].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
        default=Path(__file__).parent / 'outputs' / 'demo_gpu' / 'checkpoint.pt')
    parser.add_argument('--prompt', default='o modelo ')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    print(generate_from_checkpoint(args.checkpoint, args.prompt, args.tokens,
                                    args.temperature, args.device, args.seed))
