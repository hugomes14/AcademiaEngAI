# pip install -U diffusers

import torch
from diffusers import StableDiffusionPipeline

# Classic Art Movements
pt_01_baroque_style = "A lavish scene painted in the ornate Baroque style, deep chiaroscuro, dramatic contrasts of light and shadow, and rich detail."
# "Um cenário luxuoso pintado no estilo ornamentado barroco, com profundos contrastes de claro-escuro, contrastes dramáticos entre luz e sombra, e ricos detalhes."

pt_02_impressionist_style = "A vibrant Impressionist landscape with loose brushstrokes, soft color transitions, and natural lighting reminiscent of a Monet painting."
# "Uma paisagem impressionista vibrante, com pinceladas soltas, transições suaves de cor e iluminação natural, evocando uma pintura de Monet."

pt_03_cubist_style = "A Cubist portrait with geometric fragmentation, overlapping planes, and a muted pastel color palette."
# "Um retrato cubista com fragmentação geométrica, planos sobrepostos e uma paleta de cores pastel suaves."

# Contemporary and Digital Art Styles
pt_04_synthwave_style = "A neon-lit city skyline at night in a retro-futuristic Synthwave style, with pink and cyan hues and 1980s aesthetics."
# "Um horizonte urbano iluminado por néon à noite, num estilo synthwave retro-futurista, com tons de rosa e ciano e estética dos anos 80."

pt_05_cyberpunk_style = "A gritty cyberpunk alleyway bathed in neon glow, detailed linework, holographic signs, and synthetic rain reflections."
# "Um beco cyberpunk áspero banhado em brilho néon, com traços detalhados, letreiros holográficos e reflexos de chuva sintética."

pt_06_lowpoly_style = "A serene mountain landscape in a minimalist low-poly style, with simple, geometric shapes and flat color shading."
# "Uma paisagem montanhosa serena em estilo low-poly minimalista, com formas simples e geométricas e sombreamento de cor uniforme."

# Photographic and Cinematic Styles
pt_07_hyperrealistic_photography = "A hyperrealistic portrait of a person with extraordinary detail, soft studio lighting, and sharp focus, resembling a high-resolution DSLR photo."
# "Um retrato hiper-realista de uma pessoa com detalhes extraordinários, iluminação de estúdio suave e foco nítido, semelhante a uma fotografia DSLR de alta resolução."

pt_08_film_noir = "A high-contrast black-and-white city scene, with smoky streets, dramatic shadows, and a moody 1940s film noir atmosphere."
# "Uma cena urbana a preto e branco de alto contraste, com ruas enevoadas, sombras dramáticas e uma atmosfera de film noir dos anos 1940."

pt_09_wes_anderson_style = "A symmetrical interior rendered in a Wes Anderson style, with pastel color schemes, whimsical furniture, and a quirky, curated aesthetic."
# "Um interior simétrico representado ao estilo de Wes Anderson, com esquemas de cores pastel, mobiliário fantasioso e uma estética peculiar e bem cuidada."

# Illustration and Comics
pt_10_manga_style = "A dynamic action scene in a classic manga style, expressive linework, vibrant hues, and stylized character proportions."
# "Uma cena de ação dinâmica no estilo clássico de manga, com traços expressivos, cores vibrantes e proporções estilizadas das personagens."

pt_11_graphic_novel_ink = "A dramatic moment illustrated with bold black ink lines, limited color palette, and a gritty, textured feel akin to a graphic novel panel."
# "Um momento dramático ilustrado com traços marcantes de tinta preta, paleta de cores limitada e uma sensação crua e texturada semelhante a um painel de banda desenhada."

pt_12_childrens_book_watercolor = "A gentle forest scene in a children’s storybook watercolor style, with soft brushwork, warm tones, and whimsical animal characters."
# "Uma cena suave de floresta ao estilo de ilustração em aquarela de um livro infantil, com pinceladas suaves, tons quentes e personagens animais encantadores."

# Cultural and Historical Influences
pt_13_medieval_manuscript = "An ornate medieval scene drawn like a manuscript illumination, with intricate borders, gold leaf details, and flat perspective."
# "Uma cena medieval ornamentada desenhada como uma iluminura de manuscrito, com bordas intrincadas, detalhes em folha de ouro e perspectiva plana."

pt_14_art_nouveau = "A portrait framed by swirling floral motifs, elegant curvilinear lines, and pastel tones in the style of Alphonse Mucha."
# "Um retrato emoldurado por motivos florais ondulantes, linhas curvas elegantes e tons pastel, ao estilo de Alphonse Mucha."

pt_15_islamic_geometric = "A symmetrical composition with intricate Islamic geometric patterns, jewel-like colors, and careful tessellation."
# "Uma composição simétrica com padrões geométricos islâmicos intrincados, cores semelhantes a joias e uma tesselagem cuidadosa."

# Surreal and Conceptual Styles
pt_16_salvador_dali_esque = "A dreamlike desert landscape with melting clocks and floating objects painted in the style of Salvador Dalí."
# "Uma paisagem desértica onírica com relógios derretidos e objetos flutuantes, pintada ao estilo de Salvador Dalí."

pt_17_abstract_expressionism = "An energetic, abstract scene with bold brushstrokes, intense colors, and emotional splatters reminiscent of Jackson Pollock."
# "Uma cena abstrata enérgica com pinceladas arrojadas, cores intensas e salpicos emocionais que evocam Jackson Pollock."

pt_18_fantasy_concept_art = "A towering castle floating in the clouds, designed as intricate fantasy concept art, with painterly details and ethereal lighting."
# "Um castelo imponente a flutuar nas nuvens, concebido como arte conceptual de fantasia, com detalhes pictóricos e iluminação etérea."

# Modern Design and Minimalism
pt_19_flat_vector = "A flat, vector-style illustration of a city skyline, simple shapes, clean lines, and a restricted, modern color palette."
# "Uma ilustração de horizonte urbano em estilo vetorial plano, com formas simples, linhas limpas e uma paleta de cores moderna e limitada."

pt_20_bauhaus_poster = "A minimalistic, geometric composition with primary colors and clean typography, influenced by Bauhaus design principles."
# "Uma composição minimalista e geométrica com cores primárias e tipografia limpa, influenciada pelos princípios do design Bauhaus."

pt_21_mid_century_modern = "A lounge interior scene with warm earthy tones, sleek furniture, and geometric patterns inspired by mid-century modern design."
# "Uma cena de interior de salão com tons terrosos e quentes, mobiliário elegante e padrões geométricos, inspirada no design moderno de meados do século."

#############################
pt_23 = "Medium shot of a young chef at work in a bustling open kitchen, carefully garnishing a dish under warm, diffuse lighting. Stainless steel countertops, vibrant ingredients, and the subtle clatter of utensils in the background convey a sense of culinary artistry and focused precision."
# "Plano médio de um jovem chef a trabalhar numa cozinha aberta e movimentada, a decorar cuidadosamente um prato sob uma iluminação quente e difusa. Bancadas em aço inoxidável, ingredientes vibrantes e o tilintar subtil de utensílios ao fundo transmitem um sentido de arte culinária e precisão focada."

pt_24 = "Aerial shot of a busy farmers’ market in an old town square at midday, stalls overflowing with fresh produce, flowers, and handmade crafts. Shoppers meander between colorful umbrellas, chatting and haggling, while a nearby fountain softly trickles, adding a lively, communal energy to the scene."
# "Plano aéreo de um mercado de agricultores movimentado numa antiga praça da cidade ao meio-dia, com bancas repletas de produtos frescos, flores e artesanato. Os clientes percorrem as sombrinhas coloridas, conversando e regateando, enquanto uma fonte próxima goteja suavemente, acrescentando uma energia viva e comunitária ao cenário."

pt_25 = "Close-up of a ballet dancer leaning against the backstage wall of a dimly lit theater, her face illuminated by a single spotlight filtering through the heavy velvet curtains. Dust motes hover in the glow, and the quiet hum of distant music captures a moment of poised anticipation before her grand performance."
# "Plano aproximado de uma bailarina encostada à parede nos bastidores de um teatro com iluminação ténue, o seu rosto iluminado por um único foco de luz filtrado através das pesadas cortinas de veludo. Partículas de pó flutuam no brilho, e o leve zumbido de música distante capta um momento de antecipação serena antes da sua grande atuação."

# Base prompt
prompt = pt_13_medieval_manuscript


pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype = torch.float16
).to("cuda:0")

image = pipe(
    prompt          = prompt,
    generator       = torch.Generator("cuda:0").manual_seed(6),
    width           = 768,
    height          = 512
).images[0]

image.save("imagem_ex4.jpg")
