"""Contratos do Transformer: ver o futuro deve ser impossível."""
import unittest

import torch

from model import MiniTransformer, ModelConfig


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.config = ModelConfig(11, context_size=8, d_model=16,
                                  n_heads=4, n_layers=2, dropout=0.0)
        self.model = MiniTransformer(self.config).eval()

    def test_changing_future_tokens_cannot_change_prefix_predictions(self):
        original = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 9, 8]])
        a, _ = self.model(original)
        b, _ = self.model(changed)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-7)

    def test_attention_is_normalized_and_never_looks_ahead(self):
        logits, attention = self.model(torch.tensor([[1, 2, 3, 4]]),
                                       capture_attention=True)
        self.assertEqual(tuple(logits.shape), (1, 4, 11))
        self.assertEqual(len(attention), 2)
        for weights in attention:
            self.assertEqual(tuple(weights.shape), (1, 4, 4, 4))
            self.assertEqual(torch.count_nonzero(torch.triu(weights, 1)).item(), 0)
            torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4, 4))
            torch.testing.assert_close(weights[0, :, 0, 0], torch.ones(4))

    def test_generation_keeps_prompt_when_context_window_is_exceeded(self):
        prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        result = self.model.generate(prompt, max_new_tokens=5)
        self.assertEqual(tuple(result.shape), (1, 14))
        torch.testing.assert_close(result[:, :9], prompt)

    def test_all_trainable_parameters_receive_finite_gradients(self):
        self.model.train()
        logits, _ = self.model(torch.tensor([[1, 2, 3, 4]]))
        torch.nn.functional.cross_entropy(logits.reshape(-1, 11),
                                           torch.tensor([2, 3, 4, 5])).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_architectures_and_generation_parameters_are_rejected(self):
        for kwargs in ({'d_model': 15, 'n_heads': 4}, {'context_size': 0},
                       {'n_layers': 0}, {'vocab_size': 1}, {'dropout': 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**({'vocab_size': 11} | kwargs))
        for kwargs in ({'temperature': 0}, {'max_new_tokens': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.generate(torch.tensor([[1]]), **kwargs)

    def test_empty_and_overlong_inputs_are_rejected(self):
        for tokens in (torch.empty((1, 0), dtype=torch.long),
                       torch.ones((1, 9), dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.model(tokens)


if __name__ == '__main__':
    unittest.main()
