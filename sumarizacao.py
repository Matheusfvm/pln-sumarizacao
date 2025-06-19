# Tokenizer 
from transformers import T5Tokenizer
import re
# PyTorch model 
from transformers import T5Model, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

token_name = 'unicamp-dl/ptt5-base-portuguese-vocab'
model_name = 'recogna-nlp/ptt5-base-summ'
tokenizer = T5Tokenizer.from_pretrained(token_name )
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = '''
O furacão Erick chegou ao México com ventos de mais de 200 km/h, informou o Centro Nacional de Furacões dos EUA (NHC, na sigla em inglês) na manhã desta quinta-feira (19). O fenômeno tocou o solo no estado de Oaxaca, na costa do Pacífico no sul do país, como um furacão de categoria 3.

O NHC afirmou que o furacão Erick atualmente tem "ventos destruidores" de até 205 km/h, reduzindo sua categoria de 4 para 3, e se move na direção noroeste a uma velocidade de 15 km/h. Agora que tocou o solo, as estimativas são que o fenômeno deva perder força rapidamente por uma região montanhosa do México e se dissipar já na sexta-feira, segundo o NHC.

Ainda segundo o Centro Nacional de Furacões americano, o furacão Erick deve causar chuvas de até 40 cm e enchentes potencialmente letais nos estados de Oaxaca e Guerrero, além de inundações em áreas costeiras igualmente perigosas. A cidade turística de Acapulco fica em Guerrero.
Não há relato de feridos ou de mortos pelo furacão Erick até a última atualização desta reportagem. Pouco antes de chegar ao México, o furacão era de categoria 4, com ventos de 230 km/h e foi chamado de "extremamente perigoso" pela NHC.

Um furacão é definido como de grande porte quando é de categoria 3 ou superior e com ventos de pelo menos 180 km/h. Os meteorologistas previam um fortalecimento adicional e disseram que danos devastadores por ventos eram possíveis nas áreas próximas ao ponto de impacto do olho da tempestade.

No fim da quarta-feira, a trajetória projetada do Erick avançou mais ao sul, aproximando-se da cidade turística de Puerto Escondido, no estado de Oaxaca, e centralizando-se em um trecho pouco povoado da costa, entre Puerto Escondido e Acapulco, ao noroeste.

A presidente Claudia Sheinbaum afirmou, em uma mensagem em vídeo na noite de quarta-feira, que todas as atividades na região foram suspensas e pediu que as pessoas permanecessem em suas casas ou buscassem abrigos, caso morassem em áreas de risco.

Ao anoitecer, ondas já invadiam a orla de Puerto Escondido, alagando barcos de pesca que haviam sido puxados para a areia por segurança. A praia desapareceu sob as ondas violentas e a maré alta já alcançava o interior de alguns restaurantes à beira-mar.

As compras de última hora terminaram ao cair da noite, com lojas fechando e as ruas ficando desertas.

Mais cedo, pescadores de Puerto Escondido retiraram seus barcos da água em preparação para a chegada da tempestade. Alguns surfistas ainda se arriscavam nas ondas da praia de Zicatela, mesmo com as bandeiras vermelhas de alerta.

A mudança na rota da tempestade foi vista como um possível alívio para os moradores de Acapulco, já castigada por furacões.

A cidade de quase 1 milhão de habitantes foi devastada em outubro de 2023 pelo furacão Otis, de categoria 5, que se intensificou rapidamente e pegou muitos de surpresa. Pelo menos 52 pessoas morreram e praticamente todos os hotéis do resort sofreram danos severos.
'''

def remover_tags(texto):
    texto_limpo = re.sub(r'</?s>|<pad>', '', texto)
    return texto_limpo.strip()

def contar_palavras(texto):
    palavras = texto.split()
    return len(palavras)

print("Quantidade de palavras do texto original: " + str(contar_palavras(text)))
inputs = tokenizer.encode(text, max_length=1024, truncation=True, return_tensors='pt')
summary_ids = model.generate(inputs, max_length=512, min_length=150, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
summary = tokenizer.decode(summary_ids[0])
summary = remover_tags(summary)
print("Quantidade de palavrs do resumo: " + str(contar_palavras(summary)))
print("Resumo: \n" + summary)
