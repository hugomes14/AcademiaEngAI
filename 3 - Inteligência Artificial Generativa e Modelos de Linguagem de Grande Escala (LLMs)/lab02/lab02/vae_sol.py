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
def set_seed(seed_value=42):
    """
    Define a semente para reprodutibilidade.
    Garante que as mesmas operações produzam 
    resultados consistentes em cada execução.
    """
    # Seed para o NumPy
    np.random.seed(seed_value)
    # Seed para PyTorch (CPU)
    torch.manual_seed(seed_value)
    # Seed para PyTorch (GPU)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # se estiveres a utilizar múltiplas GPUs
    # Seed para a biblioteca 'random'
    random.seed(seed_value)
    # Seed para hashing de strings (Python)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Comportamento determinístico em certas operações CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Chama a função para fixar a semente
set_seed()

# Verifica se há GPU disponível; caso contrário, usa CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ====================================================
# Ex04: Carregar o conjunto de dados MNIST
# ====================================================
# MNIST: dígitos (0-9) escritos à mão. Cada imagem: 28x28 px.
# transforms.ToTensor() converte a imagem em tensores PyTorch (valores entre [0,1]).

transform = transforms.Compose([transforms.ToTensor()])

# Carrega os dados de treino
trainset = torchvision.datasets.MNIST(
    root='./data',       # diretoria onde os dados serão guardados
    train=True,          # True = conjunto de treino
    download=True,       # se ainda não existir, faz download automaticamente
    transform=transform  # aplica ToTensor
)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

# Carrega os dados de teste (para avaliação)
testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,         # False = conjunto de teste
    download=True,
    transform=transform
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# ====================================================
# Ex05: Definir o modelo VAE
# ====================================================
class VAE(nn.Module):
    """
    Variational Autoencoder composto por:
    - Encoder: mapeia entrada para mu, logvar (distribuição latente).
    - Decoder: reconstrói a imagem a partir de um vetor latente z.
    """
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        # fc1: transforma 28*28 entradas em 400 neurónios
        self.fc1 = nn.Linear(28 * 28, 400)
        # fc21: produz o valor de média (mu) para cada dimensão do espaço latente
        self.fc21 = nn.Linear(400, latent_dim)
        # fc22: produz o valor de log-variância (logvar) para cada dimensão do espaço latente
        self.fc22 = nn.Linear(400, latent_dim)

        # Decoder
        # fc3: recebe um vetor de dimensão latente e produz 400 neurónios
        self.fc3 = nn.Linear(latent_dim, 400)
        # fc4: reconstrói 28*28 valores (uma imagem achada)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        """
        Aplica fc1 + ReLU, devolvendo:
        - mu: média do espaço latente
        - logvar: log da variância do espaço latente
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + eps * sigma,
        onde sigma = exp(0.5 * logvar).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   # gera ruído com mesma forma de std
        return mu + eps * std

    def decode(self, z):
        """
        Decodifica o vetor z para gerar a imagem reconstruída.
        """
        h3 = F.relu(self.fc3(z))
        # usa sigmoid para manter valores de pixel entre [0,1]
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """
        Fluxo de dados:
        1) Entra x
        2) Encode -> mu, logvar
        3) Reparameteriza -> z
        4) Decode -> recon_x
        """
        # x.view(-1, 28*28) achata a imagem para um vector 1D
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ====================================================
# Ex06: Definir a função de loss (Beta-VAE)
# ====================================================
# VAE Loss = Reconstruction Loss + beta * KL Divergence.
beta = 1.0

def loss_function(recon_x, x, mu, logvar):
    """
    Calcula a perda total do VAE:
    - BCE: Binary Cross Entropy entre recon_x e x
    - KLD: Kullback-Leibler Divergence 
    """
    # BCE: mede a diferença entre a reconstrução e o original
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, 28 * 28), reduction='sum'
    )
    # KLD: força (mu, logvar) a aproximarem-se de (0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Beta controla a ênfase do termo KLD
    return BCE + beta * KLD

# ====================================================
# Ex07: Treinar o VAE com espaço latente de dimensão 2
# ====================================================
latent_dim = 2                # Dimensão do espaço latente
model = VAE(latent_dim=latent_dim).to(device)  # Instancia modelo VAE
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Otimizador Adam
num_epochs = 100              # Número de épocas de treino

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data, _ in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        # Calcula a perda
        loss = loss_function(recon_batch, data, mu, logvar)
        # Backpropagation
        loss.backward()
        # Acumula o valor da perda para estatísticas
        train_loss += loss.item()
        # Atualiza os parâmetros do modelo
        optimizer.step()
    
    # Imprime a loss média por amostra
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(trainloader.dataset)}')

# ====================================================
# Ex08: Visualização do Espaço Latente (dim=2)
# ====================================================
model.eval()  # modo de avaliação (sem gradientes)
all_z = []
all_labels = []

with torch.no_grad():
    # Percorre o testloader e projecta cada batch no espaço latente
    for data, labels in testloader:
        data = data.to(device)
        # Extrai mu e logvar
        mu, logvar = model.encode(data.view(-1, 28 * 28))
        # Reparameteriza para obter z
        z = model.reparameterize(mu, logvar)
        # Move para CPU para conversão em numpy
        z = z.cpu().numpy()

        all_z.append(z)
        all_labels.append(labels)

    # Concatena todos os lotes
    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)

    # Plota os pontos z no espaço 2D, colorindo com base no rótulo (0-9)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1],
                          c=all_labels, cmap='tab10',
                          alpha=0.5, edgecolors='k', s=20)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('Dimensão Latente 1')
    plt.ylabel('Dimensão Latente 2')
    plt.title('Visualização do Espaço Latente (2D)')
    plt.grid(True)
    plt.show()

# ====================================================
# Ex09: Visualização de Ajuste Contínuo de Valores
#       (Continuous Value Adjustment)
# ====================================================
with torch.no_grad():
    # Definimos uma grelha de valores no espaço latente
    grid_x = np.linspace(-3, 3, 20)
    grid_y = np.linspace(-3, 3, 20)
    # Vamos criar uma imagem gigante de 28*20 x 28*20
    figure = np.zeros((28 * 20, 28 * 20))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Cria um ponto (xi, yi) no espaço latente
            z_sample = torch.Tensor([[xi, yi]]).to(device)
            # Decodifica esse vetor para obter a imagem
            x_decoded = model.decode(z_sample)
            # Molda para (28, 28) e converte para numpy
            digit = x_decoded.view(28, 28).cpu().numpy()
            # Posiciona na grelha
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit

    # Mostra a grelha final
    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='gray')
    plt.title('Visualização de Ajuste Contínuo de Valores (Latent Dim=2)')
    plt.show()

# ====================================================
# Ex10: Treinar e visualizar a reconstrução 
#       com dimensão latente 128
# ====================================================
latent_dim = 128
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data, _ in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(trainloader.dataset)}')

# Após o treino, avaliamos a reconstrução em algumas imagens de teste
model.eval()
with torch.no_grad():
    # Retira um lote do testloader
    data, _ = next(iter(testloader))
    data = data.to(device)
    # Reconstrói
    recon, _, _ = model(data)
    recon = recon.view(-1, 1, 28, 28).cpu().numpy()

    # Visualiza algumas imagens originais vs. reconstruídas
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        # Primeira linha: imagens originais
        axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Segunda linha: imagens reconstruídas
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].set_title("Reconstruído")
        axes[1, i].axis('off')

    plt.show()
