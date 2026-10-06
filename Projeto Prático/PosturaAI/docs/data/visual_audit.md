# Revisão visual das anotações SRKD

**Estado:** pendente. A geração das imagens foi concluída; a revisão manual das 100 imagens exigida pelo gate M0 ainda não foi concluída.

Em 2026-09-17, `python3 -m scripts.visualize_annotations --samples 100` produziu 100 overlays e `outputs/visualizations/ground_truth/sample_list.csv` com seed 42. Há 10 imagens por intervalo do dataset; 9 foram escolhidas por terem pés junto ao limite inferior. Na amostra, 1 box e 4 pontos visíveis ultrapassam a imagem bruta. SHA-256 do CSV: `157e0e4986f3253e5658329371c7f17e78a36ce94db47ac9949df53d43392da6`.

Os overlays `image_000410.jpg`, `image_002087.jpg`, `image_020891.jpg`, `image_033945.jpg`, `image_064721.jpg` e `image_085217.jpg` foram inspecionados numa verificação inicial sem erro óbvio de renderização. `image_033945.jpg` mostra pés abaixo do enquadramento, coerente com a anomalia registada no JSON bruto. Isto **não** aprova o mapeamento anatómico nem o skeleton. Para fechar a auditoria, rever os 100 overlays, documentar lado esquerdo/direito, ligações dos pés, pontos fora da imagem e qualquer caso duvidoso; registar o responsável e a data antes de marcar M0 como aprovado.
