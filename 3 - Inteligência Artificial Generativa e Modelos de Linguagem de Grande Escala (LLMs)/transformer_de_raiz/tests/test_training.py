"""Testes com dados reais em memória, sem simular o treino."""
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import MiniTransformer, ModelConfig
from train import (CharacterTokenizer, build_demo_corpus, make_batch,
                   resolve_device, run_training, split_corpus)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_character_tokenizer_round_trips_accents(self):
        tokenizer = CharacterTokenizer('atenção!\n')
        self.assertEqual(tokenizer.decode(tokenizer.encode('ação!')), 'ação!')
        with self.assertRaisesRegex(ValueError, 'vocabulário'):
            tokenizer.encode('z')

    def test_target_is_the_next_character_for_every_input_position(self):
        x, y = make_batch(torch.arange(20), 4, 5,
                          torch.Generator().manual_seed(42))
        self.assertEqual(tuple(x.shape), (4, 5))
        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])

    def test_short_corpus_and_invalid_batch_are_rejected(self):
        with self.assertRaises(ValueError):
            split_corpus('demasiado curto', 32)
        with self.assertRaises(ValueError):
            make_batch(torch.arange(5), 2, 5, torch.Generator())
        with self.assertRaises(ValueError):
            make_batch(torch.arange(20), 0, 5, torch.Generator())

    def test_demo_validation_has_distinct_sentences_and_known_characters(self):
        text = build_demo_corpus()
        train_text, val_text = split_corpus(text, 64)
        self.assertFalse(set(train_text.splitlines()) & set(val_text.splitlines()))
        self.assertTrue(set(val_text) <= set(train_text))
        self.assertEqual(text, build_demo_corpus())

    def test_cuda_request_never_silently_falls_back_to_cpu(self):
        if torch.cuda.is_available():
            self.assertEqual(resolve_device('cuda').type, 'cuda')
        else:
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                resolve_device('cuda')
        self.assertEqual(resolve_device('cpu').type, 'cpu')

    def test_model_can_learn_a_repeating_sequence(self):
        # Remover optimizer.step, deslocar alvos ou cortar gradientes quebra isto.
        model = MiniTransformer(ModelConfig(4, context_size=8, d_model=16,
                                             n_heads=2, n_layers=1, dropout=0))
        sequence = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(8, 1)
        targets = torch.tensor([[1, 2, 3, 0, 1, 2, 3, 0]]).repeat(8, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = None
        for _ in range(80):
            logits, _ = model(sequence)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4),
                                                       targets.reshape(-1))
            if initial is None:
                initial = loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.15)
        self.assertLess(loss.item(), initial / 5)

    def test_short_run_exports_metrics_report_and_loadable_checkpoint(self):
        scratch = Path(__file__).resolve().parent.parent / 'outputs'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            args = argparse.Namespace(steps=2, eval_every=1, eval_batches=1,
                batch_size=2, context_size=8, d_model=16, heads=2, layers=1,
                dropout=0.0, lr=0.003, seed=42, device='cpu',
                output=Path(folder), corpus=None, sample_tokens=8)
            summary = run_training(args)
            self.assertEqual(summary['device_type'], 'cpu')
            for filename in ('metrics.csv', 'samples.json', 'summary.json',
                             'checkpoint.pt', 'attention.npz', 'relatorio.html',
                             'loss.png', 'attention.png', 'architecture.png'):
                self.assertTrue((Path(folder) / filename).is_file(), filename)
            samples = json.loads((Path(folder) / 'samples.json').read_text('utf-8'))
            self.assertEqual([s['step'] for s in samples], [0, 1, 2])
            checkpoint = torch.load(Path(folder) / 'checkpoint.pt',
                                    map_location='cpu', weights_only=True)
            restored = MiniTransformer(ModelConfig(**checkpoint['config'])).eval()
            restored.load_state_dict(checkpoint['model'])
            tokens = torch.tensor([checkpoint['probe_tokens']])
            actual, _ = restored(tokens)
            torch.testing.assert_close(actual, checkpoint['probe_logits'])
            self.assertTrue(math.isfinite(summary['final_val_loss']))
            from generate import generate_from_checkpoint
            text = generate_from_checkpoint(Path(folder) / 'checkpoint.pt',
                'o modelo ', max_new_tokens=5, device_name='cpu', seed=7)
            self.assertTrue(text.startswith('o modelo '))
            self.assertEqual(len(text), len('o modelo ') + 5)


if __name__ == '__main__':
    unittest.main()
