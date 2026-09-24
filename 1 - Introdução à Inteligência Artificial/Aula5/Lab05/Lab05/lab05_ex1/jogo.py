"""
🎭 ESCOLA DE DETETIVES: O MISTÉRIO DO CONCURSO 🔍
Um jogo interativo para descobrir porque a validação é importante!
"""

import time
import random

class JogoDetetive:
    def __init__(self):
        self.teoria_escolhida = None
        self.teoria_texto = None
        
        # Dados dos animais que JÁ competiram (TREINO)
        self.animais_passado = [
            {"nome": "Rex 🐕", "cor": "castanho", "tamanho": "grande", "truque": "SIM", "venceu": "✅"},
            {"nome": "Mimi 🐈", "cor": "branco", "tamanho": "pequeno", "truque": "SIM", "venceu": "✅"},
            {"nome": "Zeca 🐢", "cor": "verde", "tamanho": "pequeno", "truque": "NÃO", "venceu": "❌"},
            {"nome": "Piu 🦜", "cor": "colorido", "tamanho": "pequeno", "truque": "SIM", "venceu": "✅"},
            {"nome": "Bolt 🐇", "cor": "branco", "tamanho": "pequeno", "truque": "SIM", "venceu": "✅"},
            {"nome": "Luna 🦆", "cor": "amarelo", "tamanho": "pequeno", "truque": "NÃO", "venceu": "❌"},
            {"nome": "Simba 🦁", "cor": "dourado", "tamanho": "grande", "truque": "NÃO", "venceu": "✅"},
            {"nome": "Kiko 🐒", "cor": "castanho", "tamanho": "pequeno", "truque": "SIM", "venceu": "✅"},
            {"nome": "Pingo 🐧", "cor": "preto", "tamanho": "pequeno", "truque": "NÃO", "venceu": "✅"},
            {"nome": "Filó 🦎", "cor": "verde", "tamanho": "pequeno", "truque": "NÃO", "venceu": "❌"},
        ]
        
        # Novos competidores (VALIDAÇÃO)
        self.novos_competidores = [
            {"nome": "Dumbo 🐘", "cor": "cinzento", "tamanho": "GIGANTE", "truque": "SIM", "venceu": "✅"},
            {"nome": "Nemo 🐠", "cor": "laranja", "tamanho": "minúsculo", "truque": "NÃO", "venceu": "❌"},
            {"nome": "Boing 🦘", "cor": "castanho", "tamanho": "grande", "truque": "SIM", "venceu": "✅"},
            {"nome": "Joli 🐴", "cor": "castanho", "tamanho": "grande", "truque": "NÃO", "venceu": "✅"},
            {"nome": "Flip 🐬", "cor": "cinzento", "tamanho": "grande", "truque": "SIM", "venceu": "✅"},
        ]
        
        self.teorias_disponiveis = {
            'A': "Animais PEQUENOS vencem",
            'B': "Animais que fazem TRUQUES vencem",
            'C': "Animais GRANDES vencem",
            'D': "É ALEATÓRIO"
        }
    
    def limpar(self):
        print("\n" * 50)
    
    def pausa(self):
        input("\n⏸️  [Pressiona ENTER]")
    
    def mostrar_cabecalho(self, fase):
        """Mostra sempre a tabela e a teoria escolhida no topo"""
        print("="*70)
        print(f"  {fase}")
        print("="*70)
        
        # Mostrar tabela dos dados de treino
        print("\n📊 DADOS DO CONCURSO PASSADO:")
        print("┌─────────────┬──────────┬─────────┬───────────┬──────────┐")
        print("│ Animal      │ Cor      │ Tamanho │ Fez truque│ Resultado│")
        print("├─────────────┼──────────┼─────────┼───────────┼──────────┤")
        
        for animal in self.animais_passado:
            print(f"│ {animal['nome']:10} │ {animal['cor']:8} │ {animal['tamanho']:7} │ {animal['truque']:9} │ {animal['venceu']:7} │")
        
        print("└─────────────┴──────────┴─────────┴───────────┴──────────┘")
        
        # Mostrar teoria escolhida se já existir
        if self.teoria_texto:
            print(f"\n💡 A TUA TEORIA: {self.teoria_texto}")
        
        print("\n" + "─"*70 + "\n")
    
    def mostrar_intro(self):
        self.limpar()
        print("\n" + "="*70)
        print("  🔍 ESCOLA DE DETETIVES 🔍")
        print("="*70 + "\n")
        
        print("👋 Olá, detetive!")
        print("\n📺 NOTÍCIA: Houve um concurso de talentos de animais!")
        print("🤔 O júri usou uma REGRA SECRETA para escolher os vencedores.")
        print("\n🎯 A TUA MISSÃO:")
        print("   1️⃣  Investigar os concorrentes do passado")
        print("   2️⃣  Descobrir a regra secreta")
        print("   3️⃣  Prever quem vai ganhar no PRÓXIMO concurso")
        print("\n🏆 Vamos ver se és bom detetive!")
        
        self.pausa()
    
    def mostrar_tabela_passado(self):
        self.limpar()
        self.mostrar_cabecalho("📋 PASSO 1: INVESTIGAÇÃO")
        
        print("🔍 Observa bem a tabela acima!")
        print("   Que padrão vês? O que faz um animal VENCER?")
        
        self.pausa()
    
    def fase_teorias(self):
        self.limpar()
        self.mostrar_cabecalho("🧠 PASSO 2: CRIAR TEORIAS")
        
        print("Baseado nos dados acima, qual ACHAS que é a regra?")
        print("\nEscolhe uma teoria:\n")
        
        for letra, teoria in self.teorias_disponiveis.items():
            print(f"   {letra}) {teoria}")
        
        while True:
            escolha = input("\n🤔 A tua teoria é (A/B/C/D)? ").upper().strip()
            if escolha in ['A', 'B', 'C', 'D']:
                self.teoria_escolhida = escolha
                self.teoria_texto = self.teorias_disponiveis[escolha]
                break
            print("❌ Escolhe A, B, C ou D!")
        
        return escolha
    
    def verificar_teoria_nos_dados_passado(self, teoria):
        self.limpar()
        self.mostrar_cabecalho("🧪 PASSO 3: TESTAR A TEORIA")
        
        print("🔎 Vamos verificar se a tua teoria funciona nos dados antigos...\n")
        
        acertos = 0
        total = len(self.animais_passado)
        
        for animal in self.animais_passado:
            # Fazer previsão baseada na teoria
            if teoria == 'A':
                previsao = animal['tamanho'] == 'pequeno'
            elif teoria == 'B':
                previsao = animal['truque'] == 'SIM'
            elif teoria == 'C':
                previsao = animal['tamanho'] == 'grande'
            elif teoria == 'D':
                previsao = random.choice([True, False])  # Aleatório
            
            real = animal['venceu'] == '✅'
            correto = previsao == real
            
            if correto:
                acertos += 1
                simbolo = "✅"
            else:
                simbolo = "❌"
            
            print(f"{simbolo} {animal['nome']} - {'Acertaste!' if correto else 'Falhaste!'}")
            time.sleep(0.3)
        
        percentagem = (acertos / total) * 100
        print(f"\n📊 Resultado: {acertos}/{total} ({percentagem:.0f}%)")
        
        if percentagem == 100:
            print("🎉 UAU! A tua teoria funcionou PERFEITAMENTE nos dados antigos!")
            print("😎 Estás confiante? Então vamos testar com animais NOVOS!")
        elif percentagem >= 50:
            print("🤔 A tua teoria funcionou mais ou menos...")
            print("💭 Mas vamos testar com animais novos para ter certeza!")
        else:
            print("😅 Hmm... a tua teoria não funcionou bem...")
            print("🔄 Mas vamos na mesma testar com animais novos!")
        
        self.pausa()
        return teoria
    
    def fase_predicao(self, teoria):
        self.limpar()
        self.mostrar_cabecalho("🎪 PASSO 4: O NOVO CONCURSO!")
        
        print("\n📢 CHEGOU O DIA! Há 5 novos competidores!")
        print("🎯 Usa a tua teoria para prever quem vai vencer!\n")
        
        predicoes = []
        
        for i, animal in enumerate(self.novos_competidores, 1):
            print(f"{'─'*70}")
            print(f"🎭 COMPETIDOR {i}: {animal['nome']}")
            print(f"   📝 Cor: {animal['cor']}")
            print(f"   📏 Tamanho: {animal['tamanho']}")
            print(f"   🎪 Faz truque: {animal['truque']}")
            
            while True:
                palpite = input(f"\n   🤔 Este animal vai VENCER? (S/N): ").upper().strip()
                if palpite in ['S', 'N']:
                    predicoes.append(palpite == 'S')
                    if palpite == 'S':
                        print("   ✓ Apostas que vai VENCER! 🏆")
                    else:
                        print("   ✓ Apostas que vai PERDER! ❌")
                    break
                print("   ⚠️  Responde S (sim) ou N (não)!")
            
            print()
            time.sleep(0.3)
        
        return predicoes
    
    def revelar_resultados(self, teoria, predicoes):
        self.limpar()
        self.mostrar_cabecalho("🎊 MOMENTO DA VERDADE!")
        
        print("🥁 Os resultados estão a ser revelados...\n")
        time.sleep(1)
        
        acertos = 0
        
        for i, animal in enumerate(self.novos_competidores):
            print(f"{'─'*70}")
            print(f"🎭 {animal['nome']}")
            
            resultado_real = animal['venceu'] == '✅'
            tua_previsao = predicoes[i]
            
            print(f"   🔮 Tu previste: {'VENCER 🏆' if tua_previsao else 'PERDER ❌'}")
            time.sleep(1)
            print(f"   🎯 Resultado real: {'VENCEU 🏆' if resultado_real else 'PERDEU ❌'}")
            time.sleep(0.5)
            
            if tua_previsao == resultado_real:
                print(f"   ✅ ACERTASTE!")
                acertos += 1
            else:
                print(f"   ❌ ERRASTE!")
            
            print()
            time.sleep(0.8)
        
        total = len(self.novos_competidores)
        percentagem = (acertos / total) * 100
        
        print(f"{'='*70}")
        print(f"📊 PONTUAÇÃO FINAL: {acertos}/{total} ({percentagem:.0f}%)")
        print(f"{'='*70}")
        
        if percentagem == 100:
            print("\n🎉 INCRÍVEL! Acertaste tudo!")
        elif percentagem >= 66:
            print("\n😊 Não foi mau! Acertaste a maioria!")
        else:
            print("\n😅 Hmm... não correu muito bem...")
        
        self.pausa()
        return acertos, total
    
    def grande_revelacao(self, teoria, acertos, total):
        self.limpar()
        self.mostrar_cabecalho("🎬 A GRANDE REVELAÇÃO!")
        
        print("🔍 Detetive, tenho algo importante para te contar...\n")
        time.sleep(1)
        
        print("🎯 A regra SECRETA do júri era:")
        time.sleep(1)
        print("\n" + "="*70)
        print("   🎪 Se o animal FEZ TRUQUES → VENCE ✅")
        print("   🎪 Se o animal NÃO FEZ TRUQUES → PERDE ❌")
        print("="*70 + "\n")
        time.sleep(1)
        
        if teoria == 'B':
            print("🎉 PARABÉNS! Descobriste a regra certa!")
            print(f"   E acertaste {acertos}/{total} previsões!")
            print("\n💡 Mas repara: alguns animais que NÃO fizeram truques também ganharam!")
            print("   (Simba 🦁, Pingo 🐧, Zé 🐴)")
            print("\n🤔 Isso mostra que a regra NÃO é 100% perfeita...")
            print("   Podem existir outros fatores! (carisma, sorte, júri subjetivo)")
        else:
            print("😮 Ops! A tua teoria estava errada...")
            print(f"\n   A tua teoria era: '{self.teoria_texto}'")
            print(f"   Mas a regra certa era: 'Animais que fazem TRUQUES vencem'")
            print("\n🤔 Mas espera... então porque é que a tua teoria")
            print("   funcionou TÃO BEM nos dados antigos?")
            time.sleep(1)
            print("\n💡 Esse é exatamente o problema!")
        
        self.pausa()
    
    def explicacao_final(self, teoria):
        self.limpar()
        self.mostrar_cabecalho("🎓 A GRANDE LIÇÃO")
        
        print("🔬 O que aprendemos:\n")
        
        if teoria != 'B':
            print("❌ A tua teoria funcionou BEM nos dados ANTIGOS")
            print("❌ Mas FALHOU com dados NOVOS\n")
            print("🤔 Porquê?")
            print("\n📊 Nos 10 animais antigos, por COINCIDÊNCIA:")
            
            if teoria == 'A':
                print("   • Muitos pequenos fizeram truques (Mimi, Piu, Bolt, Kiko)")
                print("   • ALGUNS pequenos ganharam SEM truques (Pingo 🐧)")
                print("   • Pareciam ganhar POR SEREM pequenos")
                print("   • Mas na maioria dos casos, faziam truques!")
                print("\n🐘 Quando apareceu o Dumbo (GIGANTE + TRUQUES) → VENCEU!")
                print("🐴 E o Zé (GRANDE sem truques) também venceu!")
                print("   A tua teoria 'pequenos vencem' não explica tudo!")
            
            elif teoria == 'C':
                print("   • Alguns grandes fizeram truques (Rex 🐕)")
                print("   • Outros grandes NÃO fizeram truques (Simba 🦁)")
                print("   • AMBOS ganharam!")
                print("   • Pareciam ganhar POR SEREM grandes")
                print("\n🦘 Mas outros grandes perderam também!")
                print("   O tamanho não era o fator principal!")
            
            elif teoria == 'D':
                print("   • Esta teoria assume que tudo é aleatório")
                print("   • Mas claramente HÁ um padrão!")
                print("   • A maioria dos que fazem truques ganha")
                print("   • Alguns sem truques também ganham (carisma? sorte?)")
                print("\n🎯 Não é 100% aleatório - há tendências!")
        else:
            print("✅ Parabéns! Descobriste a melhor teoria!")
            print("\n💡 Mas repara que a regra não é 100% perfeita:")
            print("   • Alguns fizeram truques e perderam? NÃO")
            print("   • Alguns NÃO fizeram truques e ganharam? SIM (Simba, Pingo, Zé)")
            print("\n🤔 Isso mostra que existem OUTROS fatores:")
            print("   • Carisma do animal")
            print("   • Humor do júri")
            print("   • Sorte do momento")
            print("   • Tipo de animal (leão impressiona mesmo sem truques!)")
            print("\n📊 A regra 'truques' é a MELHOR, mas não é perfeita!")
        
        print("\n" + "="*70)
        print("⚠️  ISTO CHAMA-SE: OVERFITTING (SOBREAJUSTE)")
        print("="*70)
        
        print("\n📚 Significa:")
        print("   ✓ O teu 'modelo' decorou os dados antigos")
        print("   ✓ Encontrou padrões que eram só COINCIDÊNCIA")
        print("   ✓ Não aprendeu a regra VERDADEIRA (ou completa)")
        print("   ✓ Por isso FALHOU com dados novos")
        print("\n💡 Lições importantes:")
        print("   • Nenhum modelo é 100% perfeito no mundo real")
        print("   • Sempre existem exceções e outros fatores")
        print("   • A validação ajuda a descobrir as limitações")
        print("   • É melhor ter 80% certo do que 100% decorado!")
        
        self.pausa()
    
    def solucao(self):
        self.limpar()
        print("\n" + "="*70)
        print("  ✅ A SOLUÇÃO: VALIDAÇÃO!")
        print("="*70 + "\n")
        
        print("🎯 Como evitar este erro?\n")
        
        print("💡 VALIDAÇÃO = Testar o modelo com dados NOVOS\n")
        
        print("📋 O que devíamos ter feito:")
        print("   1️⃣  Guardar alguns animais antigos só para TESTAR")
        print("   2️⃣  Criar a teoria com os OUTROS animais")
        print("   3️⃣  VALIDAR a teoria nos animais guardados")
        print("   4️⃣  Se falhar → teoria está errada!")
        
        print("\n🔬 Em Machine Learning:")
        print("   📊 Dados de TREINO = aprender padrões")
        print("   ✅ Dados de VALIDAÇÃO = testar se funciona")
        print("   🎯 Dados de TESTE = avaliação final")
        
        print("\n⚠️  NUNCA confies só nos dados de treino!")
        
        self.pausa()
    
    def exemplos_reais(self):
        self.limpar()
        print("\n" + "="*70)
        print("  🌍 NO MUNDO REAL")
        print("="*70 + "\n")
        
        print("💭 Imagina se isto acontecesse em situações sérias:\n")
        
        exemplos = [
            ("🏥 MEDICINA", "Um modelo prevê doenças mas só foi treinado em homens.",
             "Vai falhar em mulheres! PERIGOSO! ☠️"),
            
            ("🚗 CARROS", "Um carro autónomo só 'viu' estradas em dias de sol.",
             "Vai bater quando chover! DESASTRE! 💥"),
            
            ("💰 BANCO", "Um modelo aprova empréstimos mas tem dados enviesados.",
             "Vai discriminar pessoas! INJUSTO! ⚖️"),
        ]
        
        for titulo, situacao, consequencia in exemplos:
            print(f"{titulo}")
            print(f"   📝 {situacao}")
            print(f"   ⚠️  {consequencia}\n")
            time.sleep(1.5)
        
        print("="*70)
        print("🎯 POR ISSO A VALIDAÇÃO É OBRIGATÓRIA!")
        print("="*70)
        
        self.pausa()
    
    def final(self):
        self.limpar()
        print("\n" + "="*70)
        print("  🎊 PARABÉNS, DETETIVE!")
        print("="*70 + "\n")
        
        print("🎓 Agora já sabes:")
        print("\n   ✅ O que é overfitting (sobreajuste)")
        print("   ✅ Porque dados de treino podem enganar")
        print("   ✅ Porque a validação é ESSENCIAL")
        print("   ✅ Como testar modelos corretamente")
        
        print("\n🌟 És agora um DETETIVE DE DADOS certificado!")
        print("\n💡 Lembra-te sempre:")
        print("   'Um modelo que parece perfeito pode ser")
        print("    perfeitamente INÚTIL no mundo real!'")
        
        print("\n🚀 Partilha este jogo com os teus amigos!")
        print("\n👋 Obrigado por jogares!\n")
    
    def jogar(self):
        """Função principal do jogo"""
        self.mostrar_intro()
        self.mostrar_tabela_passado()
        teoria = self.fase_teorias()
        teoria = self.verificar_teoria_nos_dados_passado(teoria)
        predicoes = self.fase_predicao(teoria)
        acertos, total = self.revelar_resultados(teoria, predicoes)
        self.grande_revelacao(teoria, acertos, total)
        self.explicacao_final(teoria)
        self.solucao()
        self.exemplos_reais()
        self.final()

# INICIAR O JOGO
if __name__ == "__main__":
    jogo = JogoDetetive()
    jogo.jogar()