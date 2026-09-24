"""Exercício 05 — GAN simples para gerar algarismos do MNIST.

Objetivo: perceber o jogo entre duas redes, e não apenas obter imagens bonitas.

    Gerador G:       ruído z (64 números) -> imagem falsa (1 x 28 x 28)
    Discriminador D: imagem real ou falsa -> logit (antes da sigmoid)

Em cada lote fazemos duas atualizações, nesta ordem:
    1. D aprende a dar alvo 1 às imagens reais e 0 às falsas.
    2. G aprende a fazer D dar alvo 1 às suas imagens falsas.

Instalação no ambiente Python escolhido:
    python -m pip install torch torchvision matplotlib

Execução rápida para experimentar o ciclo (descarrega o MNIST na primeira vez):
    python ex05_gan_mnist.py --epochs 2 --max-batches 50

Treino mais completo:
    python ex05_gan_mnist.py --epochs 10

Para abrir os plots numa janela ao terminar:
    python ex05_gan_mnist.py --epochs 10 --show-plots

Na execução rápida, duas épocas podem ser insuficientes para formar dígitos
legíveis; servem sobretudo para observar e modificar o ciclo de treino.

As imagens, o histórico CSV, plots_treino.png e evolucao_imagens.png são
guardados em gan_mnist_saida/. Usa sempre o mesmo ruído para as imagens de
cada época: assim as diferenças mostram a evolução do modelo, e não uma
mudança das entradas.

Como ler o histórico:
    D(real) perto de 1 e D(falso) perto de 0: D distingue bem os dois grupos.
    Ambos perto de 0.5: D está indeciso; isso NÃO prova que G gera boas imagens.
    As duas perdas oscilam porque cada rede muda o problema da outra.
    Compara sempre nitidez E variedade nas grelhas; perder variedade é uma
    falha típica chamada mode collapse.

Desafios depois da primeira execução:
    A. Compara epoca_001.png com a última época. As formas ficaram mais nítidas?
    B. Experimenta --latent-dim 16 e 128. Mudam a variedade e a estabilidade?
    C. Experimenta --lr-d 0.001 mantendo --lr-g 0.0002. O que acontece aos
       valores de D(real), D(falso) e às imagens? Depois inverte as taxas.
    D. Procura repetição do mesmo algarismo em várias amostras (mode collapse).
       As perdas, sozinhas, conseguem revelar esse problema?
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Gerador(nn.Module):
    """Projeta um ponto do espaço latente numa imagem de 28 x 28 píxeis."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 28 * 28),
            nn.Tanh(),  # Saída entre -1 e 1, como as imagens reais normalizadas.
        )

    def forward(self, ruido: torch.Tensor) -> torch.Tensor:
        return self.rede(ruido).view(-1, 1, 28, 28)


class Discriminador(nn.Module):
    """Produz um logit: positivo sugere real; negativo sugere falso."""

    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            # Sem Sigmoid: BCEWithLogitsLoss já a inclui de modo mais estável.
        )

    def forward(self, imagem: torch.Tensor) -> torch.Tensor:
        return self.rede(imagem)


def treinar_lote(
    reais: torch.Tensor,
    gerador: Gerador,
    discriminador: Discriminador,
    otimizador_g: torch.optim.Optimizer,
    otimizador_d: torch.optim.Optimizer,
    criterio: nn.Module,
    latent_dim: int,
    dispositivo: torch.device,
) -> tuple[float, float, float, float]:
    """Uma jogada de D e uma de G; devolve perdas e probabilidades médias."""
    quantidade = reais.size(0)  # O último lote pode ser menor que batch-size.
    alvos_reais = torch.ones(quantidade, 1, device=dispositivo)
    alvos_falsos = torch.zeros(quantidade, 1, device=dispositivo)

    # 1) Treinar D. G cria exemplos, mas não deve aprender nesta jogada.
    # detach() corta o grafo em G: o gradiente da perda de D pára nas imagens.
    discriminador.requires_grad_(True)
    otimizador_d.zero_grad(set_to_none=True)
    logit_real = discriminador(reais)
    ruido_d = torch.randn(quantidade, latent_dim, device=dispositivo)
    falsas_para_d = gerador(ruido_d).detach()
    logit_falso = discriminador(falsas_para_d)

    perda_d_real = criterio(logit_real, alvos_reais)
    perda_d_falso = criterio(logit_falso, alvos_falsos)
    perda_d = (perda_d_real + perda_d_falso) / 2
    perda_d.backward()
    otimizador_d.step()

    # Estas probabilidades descrevem D ANTES da atualização acima.
    # São diagnósticos, não uma medida completa da qualidade visual.
    p_real = torch.sigmoid(logit_real.detach()).mean().item()
    p_falso = torch.sigmoid(logit_falso.detach()).mean().item()

    # 2) Treinar G. Queremos que D classifique as novas imagens como reais.
    # Congelamos os parâmetros de D para poupar gradientes, mas mantemos o
    # caminho D(imagem) -> imagem -> G; NÃO usamos no_grad() aqui.
    discriminador.requires_grad_(False)
    otimizador_g.zero_grad(set_to_none=True)
    ruido_g = torch.randn(quantidade, latent_dim, device=dispositivo)
    falsas_para_g = gerador(ruido_g)
    logit_para_g = discriminador(falsas_para_g)
    perda_g = criterio(logit_para_g, alvos_reais)
    # Esta é a perda "não saturante": -log(D(G(z))). Dá um sinal de treino
    # mais útil no início do que minimizar log(1 - D(G(z))).
    perda_g.backward()
    otimizador_g.step()

    return perda_d.item(), perda_g.item(), p_real, p_falso


def guardar_plot_treino(epocas: list[int], medidas: list[list[float]],
                       destino: Path) -> None:
    """Mostra separadamente perdas e respostas de D: escalas diferentes."""
    fig, (ax_perdas, ax_respostas) = plt.subplots(1, 2, figsize=(12, 4.5))
    perda_d, perda_g, p_real, p_falso = zip(*medidas)

    ax_perdas.plot(epocas, perda_d, "o-", label="Perda D")
    ax_perdas.plot(epocas, perda_g, "o-", label="Perda G")
    ax_perdas.set(title="Perdas médias", xlabel="Época", ylabel="BCE")
    ax_perdas.legend()
    ax_perdas.grid(alpha=0.3)

    ax_respostas.plot(epocas, p_real, "o-", label="D(real)")
    ax_respostas.plot(epocas, p_falso, "o-", label="D(falso)")
    ax_respostas.axhline(0.5, color="gray", linestyle="--", linewidth=1,
                         label="Indecisão (0,5)")
    ax_respostas.set(title="Resposta média de D antes da sua atualização",
                     xlabel="Época", ylabel="Probabilidade", ylim=(0, 1))
    ax_respostas.legend()
    ax_respostas.grid(alpha=0.3)

    # Com poucos pontos, mostrar apenas os números inteiros das épocas.
    for eixo in (ax_perdas, ax_respostas):
        eixo.set_xticks(epocas if len(epocas) <= 15 else
                        [epocas[0], *epocas[4::5], epocas[-1]])
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def guardar_evolucao_imagens(saida: Path, ultima_epoca: int) -> None:
    """Compara imagens reais e o MESMO ruído em três momentos do treino."""
    epocas = sorted({1, (ultima_epoca + 1) // 2, ultima_epoca})
    paineis = [("Imagens reais", saida / "reais.png")]
    paineis += [(f"Época {epoca}", saida / f"epoca_{epoca:03d}.png")
               for epoca in epocas]
    fig, eixos = plt.subplots(1, len(paineis),
                             figsize=(4.5 * len(paineis), 5))
    for eixo, (titulo, imagem) in zip(eixos, paineis):
        eixo.imshow(plt.imread(imagem), cmap="gray")
        eixo.set_title(titulo)
        eixo.axis("off")
    fig.suptitle("Variedade e nitidez: comparar visualmente, além das perdas")
    fig.tight_layout()
    fig.savefig(saida / "evolucao_imagens.png", dpi=150)
    plt.close(fig)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limita lotes por época; 0 usa todos os dados.")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr-d", type=float, default=0.0002)
    parser.add_argument("--lr-g", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "dados")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "gan_mnist_saida")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show-plots", action="store_true",
                        help="Abre os plots numa janela depois do treino.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_batches < 0:
        parser.error("epochs e batch-size devem ser positivos; max-batches >= 0.")
    if args.latent_dim < 1 or args.lr_d <= 0 or args.lr_g <= 0:
        parser.error("latent-dim e as taxas de aprendizagem devem ser positivos.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA não está disponível neste ambiente.")
    return args


def main() -> None:
    args = argumentos()
    torch.manual_seed(args.seed)
    dispositivo = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ToTensor: [0, 1]. Normalize: [0, 1] -> [-1, 1]. Tem de coincidir
    # com a escala das imagens produzidas pelo Tanh do gerador.
    transformacao = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    print("A carregar MNIST (o primeiro acesso requer ligação à Internet)...")
    dados = datasets.MNIST(args.data_dir, train=True, download=True,
                           transform=transformacao)
    lotes = DataLoader(dados, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=(dispositivo.type == "cuda"))

    # Grelha de referência: facilita comparar imagens reais e geradas.
    reais_referencia = torch.stack([dados[i][0] for i in range(64)])
    save_image((reais_referencia + 1) / 2, args.output_dir / "reais.png", nrow=8)

    gerador = Gerador(args.latent_dim).to(dispositivo)
    discriminador = Discriminador().to(dispositivo)
    criterio = nn.BCEWithLogitsLoss()
    # beta1=0.5 é uma escolha comum em GANs: reduz a inércia do Adam num
    # problema em que o adversário e, portanto, a perda, mudam a cada passo.
    otimizador_g = torch.optim.Adam(gerador.parameters(), lr=args.lr_g,
                                    betas=(0.5, 0.999))
    otimizador_d = torch.optim.Adam(discriminador.parameters(), lr=args.lr_d,
                                    betas=(0.5, 0.999))
    ruido_fixo = torch.randn(64, args.latent_dim, device=dispositivo)

    historico = args.output_dir / "historico.csv"
    epocas_plot: list[int] = []
    medidas_plot: list[list[float]] = []
    print(f"Dispositivo: {dispositivo} | imagens: {len(dados)} | saída: {args.output_dir}")
    with historico.open("w", newline="", encoding="utf-8") as ficheiro:
        escritor = csv.writer(ficheiro)
        escritor.writerow(["epoca", "perda_d", "perda_g", "D_real", "D_falso"])
        for epoca in range(1, args.epochs + 1):
            totais = [0.0, 0.0, 0.0, 0.0]
            numero_lotes = 0
            for reais, _ in lotes:
                reais = reais.to(dispositivo, non_blocking=True)
                medidas = treinar_lote(
                    reais, gerador, discriminador, otimizador_g,
                    otimizador_d, criterio, args.latent_dim, dispositivo,
                )
                totais = [a + b for a, b in zip(totais, medidas)]
                numero_lotes += 1
                if args.max_batches and numero_lotes >= args.max_batches:
                    break

            medias = [total / numero_lotes for total in totais]
            epocas_plot.append(epoca)
            medidas_plot.append(medias)
            escritor.writerow([epoca, *[f"{valor:.4f}" for valor in medias]])
            ficheiro.flush()
            print(f"Época {epoca:02d} | perda D={medias[0]:.3f} "
                  f"| perda G={medias[1]:.3f} "
                  f"| D(real)={medias[2]:.3f} | D(falso)={medias[3]:.3f}")

            # Não calculamos gradientes durante a geração das amostras.
            gerador.eval()
            with torch.no_grad():
                imagens = gerador(ruido_fixo).cpu()
            gerador.train()
            save_image((imagens + 1) / 2,
                       args.output_dir / f"epoca_{epoca:03d}.png", nrow=8)
            # O ficheiro vai sendo atualizado: pode ser consultado durante o treino.
            guardar_plot_treino(epocas_plot, medidas_plot,
                                args.output_dir / "plots_treino.png")

    guardar_evolucao_imagens(args.output_dir, args.epochs)
    torch.save({"gerador": gerador.state_dict(),
                "discriminador": discriminador.state_dict(),
                "latent_dim": args.latent_dim}, args.output_dir / "modelos.pt")
    print("Concluído. Consulta plots_treino.png e evolucao_imagens.png.")
    print("As perdas podem oscilar: qualidade e variedade vêem-se nas imagens.")
    if args.show_plots:
        for nome in ("plots_treino.png", "evolucao_imagens.png"):
            fig, eixo = plt.subplots(figsize=(12, 5))
            eixo.imshow(plt.imread(args.output_dir / nome))
            eixo.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
