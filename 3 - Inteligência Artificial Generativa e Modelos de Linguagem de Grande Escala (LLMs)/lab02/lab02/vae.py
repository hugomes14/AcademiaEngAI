# ====================================================
# Ex01: Instalar bibliotecas necessárias
#    (caso seja necessário)
#    # pip install packaging
# ====================================================

## Ex02: Importar bibliotecas necessárias
import torch                           # PyTorch para construção e treino de redes neurais
import torch.nn as nn                  # Módulo de camadas de rede neural
import torch.nn.functional as F        # Funções de ativação (ReLU, sigmoid, etc.)
import torch.optim as optim            # Otimizadores (Adam, SGD, etc.)
import numpy as np                     # NumPy para operações matriciais
import os                              # Módulo de interface com o SO
import torchvision                     # Biblioteca de visão computacional (datasets, etc.)
import torchvision.transforms as transforms  # Transformações de imagem (ToTensor, normalização, etc.)
from torch.utils.data import DataLoader       # Para carregar datasets em lotes (batches)
import matplotlib.pyplot as plt        # Para visualização de gráficos e imagens
import random                          # Biblioteca para geração de números aleatórios

# ====================================================
# Ex03: Função para definir a semente de reprodutibilidade
# ====================================================


# ====================================================
# Ex04: Carregar o conjunto de dados MNIST
# ====================================================
# MNIST: dígitos (0-9) escritos à mão. Cada imagem: 28x28 px.
# transforms.ToTensor() converte a imagem em tensores PyTorch (valores entre [0,1]).



# ====================================================
# Ex05: Definir o modelo VAE
# ====================================================


# ====================================================
# Ex06: Definir a função de loss (Beta-VAE)
# ====================================================


# ====================================================
# Ex07: Treinar o VAE com espaço latente de dimensão 2
# ====================================================


# ====================================================
# Ex08: Visualização do Espaço Latente (dim=2)
# ====================================================


# ====================================================
# Ex09: Visualização de Ajuste Contínuo de Valores
#       (Continuous Value Adjustment)
# ====================================================


# ====================================================
# Ex10: Treinar e visualizar a reconstrução 
#       com dimensão latente 128
# ====================================================

