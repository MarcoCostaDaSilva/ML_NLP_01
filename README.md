# ML_NLP_01

Bibliotecas que serão utilizadas e suas versões:
- pandas 2.1.4
- scikit-learn 1.5.2
- wordcloud 1.9.3
- matplotlib 3.7.1
- nltk 3.8.1
- unidecode 1.3.8

```python
import pandas as pd

# URL da planilha compartilhada com o link de exportação direto
url = 'https://docs.google.com/spreadsheets/d/1Irhi5XAsDqD7T2Qsl2KLUFjkUURH2UeT-3JiN-kZXpE/export?format=csv'

# Carregar a planilha como um DataFrame
df = pd.read_csv(url)

# Exibir as primeiras linhas do DataFrame
df.head()
```

```python
df.shape
```
```python
df.value_counts('sentimento')
```
```python
print('positiva \n')

df.avaliacao[0]
```
```python
print('positiva \n')

df.avaliacao[0]
```
Transformando textos em dados numéricos:

```python
from sklearn.feature_extraction.text import CountVectorizer

texto = ['Comprei um produto ótimo', 'Comprei um produto ruim']

vetorizar = CountVectorizer()
bag_of_words = vetorizar.fit_transform(texto)
```
```python
bag_of_words
```
```python
matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())
```
```python
matriz_esparsa
```

```python
vetorizar = CountVectorizer(lowercase=False)
bag_of_words = vetorizar.fit_transform(df.avaliacao)
print(bag_of_words.shape)
```
```python
vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(df.avaliacao)
print(bag_of_words.shape)
```
```python
matriz_esparsa_avaliacoes = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())
matriz_esparsa_avaliacoes
```
Classificando os sentimentos:
```python
from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words, df.sentimento, random_state=4978)
```
```python
from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_treino, y_treino)
acuracia = regressao_logistica.score(X_teste, y_teste)
print(acuracia)
```
#**Explorando a frequência e o sentimento das palavras.**

##**Visualizando as palavras mais frequentes nas avaliações**

```python
from wordcloud import WordCloud

todas_palavras = [texto for texto in df.avaliacao]

todas_palavras
```
```python
todas_palavras = ' '.join([texto for texto in df.avaliacao])

todas_palavras
```

```python
nuvem_palavras = WordCloud().generate(todas_palavras)
```
```python
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(nuvem_palavras)
plt.show()
```
![image](https://github.com/user-attachments/assets/90ae7607-9b49-49fc-a8a7-0dd107dae932)

```python
nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110).generate(todas_palavras)
plt.figure(figsize=(10,7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/ef9f74de-98b5-4074-a2a1-44f13155a65a)

```python
nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)
plt.figure(figsize=(10,7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis('off')
plt.show()
```
![image](https://github.com/user-attachments/assets/b482cc6c-d977-4683-b33f-d519b5a38a1a)

Analisando palavras por sentimento:

```python
def nuvem_palavras(texto, coluna_texto, sentimento):
  # Filtrando as resenhas com base no sentimento especificado
  texto_sentimento = texto.query(f"sentimento == '{sentimento}'")[coluna_texto]

  # Unindo todas as resenhas em uma única string
  texto_unido = ' '.join(texto_sentimento)

  # Criando e exibindo a nuvem de palavras
  nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(texto_unido)
  plt.figure(figsize=(10,7))
  plt.imshow(nuvem_palavras, interpolation='bilinear')
  plt.axis('off')
  plt.show()
```

```python
nuvem_palavras(df, 'avaliacao', 'negativo')
```
![image](https://github.com/user-attachments/assets/3e1fe0e4-65b6-4f2a-b4f5-5fd6bbea79ae)

```python
nuvem_palavras(df, 'avaliacao', 'positivo')
```
![image](https://github.com/user-attachments/assets/bfcfc5dd-b430-4dc5-994d-08b59a3e5311)

Dividindo o texto em unidades menores:

```python
todas_palavras
```
```python
import nltk

nltk.download("all")
```

```python
frases = ['um produto bom', 'um produto ruim']
frequencia = nltk.FreqDist(frases)
frequencia
```

```python
from nltk import tokenize

frase = 'O produto é excelente e a entrega foi muito rápida!'

token_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_espaco.tokenize(frase)
print(token_frase)
```
Analisando a frequência das palavras

```python
token_frase = token_espaco.tokenize(todas_palavras)

token_frase
```

```python
frequencia = nltk.FreqDist(token_frase)
frequencia

```

```python
df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()),
                              'Frequência': list(frequencia.values())})
```

```python
df_frequencia.head()
```
```python
df_frequencia.nlargest(columns='Frequência', n=10)
```
```python
import seaborn as sns

plt.figure(figsize=(20,6))
ax = sns.barplot(data=df_frequencia.nlargest(columns='Frequência', n=20), x='Palavra', y='Frequência', color='gray')
ax.set(ylabel='Contagem')
plt.show()
```
![image](https://github.com/user-attachments/assets/74d395f9-92dc-4719-9a58-a45a1f8306fb)

#**Limpando e normalizando dados textuais**

##**Removendo stopwords**

```python
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
palavras_irrelevantes
```
```python
frase_processada = []

for opiniao in df.avaliacao:
  palavras_texto = token_espaco.tokenize(opiniao)
  nova_frase = [palavra for palavra in palavras_texto if palavra not in palavras_irrelevantes]
  frase_processada.append(' '.join(nova_frase))

df['tratamento_1'] = frase_processada

df.head()
```
```python
df['avaliacao'][0]
```

```python
df['tratamento_1'][0]
```

```python
def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words, texto[coluna_classificacao], random_state=4978)
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(X_treino, y_treino)
    acuracia = regressao_logistica.score(X_teste, y_teste)
    return print(f"Acurácia do modelo com '{coluna_texto}': {acuracia * 100:.2f}%")
```

```python
classificar_texto(df, 'tratamento_1', 'sentimento')
```
```python
def grafico_frequencia(texto, coluna_texto, quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])
    token_espaco = tokenize.WhitespaceTokenizer()
    frequencia = nltk.FreqDist(token_espaco.tokenize(todas_palavras))
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                 "Frequência": list(frequencia.values())})
    df_frequencia = df_frequencia.nlargest(columns="Frequência", n=quantidade)
    plt.figure(figsize=(20,6))
    ax = sns.barplot(data=df_frequencia, x="Palavra", y ="Frequência", color='gray')
    ax.set(ylabel="Contagem")
    plt.show()
```

```python
grafico_frequencia(df, 'tratamento_1', 20)
```
![image](https://github.com/user-attachments/assets/022ef1e9-51a4-40b7-a6ed-24ced90a8ca7)

##**Removendo pontuações**:

```python
frase = 'Esse smartphone superou expectativas, recomendo'

token_pontuacao = tokenize.WordPunctTokenizer()
token_frase = token_pontuacao.tokenize(frase)
print(token_frase)
```

```python
frase_processada = []

for opiniao in df['tratamento_1']:
  palavras_texto = token_pontuacao.tokenize(opiniao)
  nova_frase = [palavra for palavra in palavras_texto if palavra.isalpha() and palavra not in palavras_irrelevantes]
  frase_processada.append(' '.join(nova_frase))

df['tratamento_2'] = frase_processada
```

```python
df.head()
```

```python
df['tratamento_1'][10]
```

```python
df['tratamento_2'][10]
```

```python
grafico_frequencia(df, 'tratamento_2', 20)
```
![image](https://github.com/user-attachments/assets/2a70fa43-3e10-451f-a895-c0c5de63721c)

Removendo acentuação:

```python
! pip install unidecode
```

```python
import unidecode

frase =  'Um aparelho ótima performance preço bem menor outros aparelhos marcas conhecidas performance semelhante'

teste = unidecode.unidecode(frase)
print(teste)
```

```python
sem_acentos = [unidecode.unidecode(texto) for texto in df['tratamento_2']]
```

```python
stopwords_sem_acento = [unidecode.unidecode(texto) for texto in palavras_irrelevantes]
```

```python
df['tratamento_3'] = sem_acentos

frase_processada = []

for opiniao in df['tratamento_3']:
  palavras_texto = token_pontuacao.tokenize(opiniao)
  nova_frase = [palavra for palavra in palavras_texto if palavra not in stopwords_sem_acento]
  frase_processada.append(' '.join(nova_frase))

df['tratamento_3'] = frase_processada
```

```python
df.head()
```

```python
df['tratamento_2'][70]
```

```python
df['tratamento_3'][70]
```

```python
grafico_frequencia(df, 'tratamento_3', 20)
```
![image](https://github.com/user-attachments/assets/da36834e-daa9-4b35-907d-6f2e99cc184e)

Uniformizando o texto:

```python
frase = 'Bom produto otimo custo beneficio Recomendo Confortavel bem acabado'
print(frase.lower())
```

```python
frase_processada = []

for opiniao in df['tratamento_3']:
  opiniao = opiniao.lower()
  palavras_texto = token_pontuacao.tokenize(opiniao)
  nova_frase = [palavra for palavra in palavras_texto if palavra not in stopwords_sem_acento]
  frase_processada.append(' '.join(nova_frase))

df['tratamento_4'] = frase_processada
```

```python
df.head()
```

```python
df['tratamento_3'][3]
```

```python
df['tratamento_4'][3]
```

```python
classificar_texto(df, 'tratamento_1', 'sentimento')
```

```python
classificar_texto(df, 'tratamento_4', 'sentimento')
```
#**Utilizando outras técnicas de processamento de texto**

##**Simplificando as palavras:**

```python
stemmer = nltk.RSLPStemmer()

stemmer.stem('gostei')
```

```python
stemmer.stem('gostado')
```
```python
stemmer.stem('gostou')
```
```python
frase_processada = []

for opiniao in df['tratamento_4']:
  palavras_texto = token_pontuacao.tokenize(opiniao)
  nova_frase = [stemmer.stem(palavra) for palavra in palavras_texto]
  frase_processada.append(' '.join(nova_frase))

df['tratamento_5'] = frase_processada
```
```python
df.head()
```
```python
df['tratamento_4'][3]
```
```python
df['tratamento_5'][3]
```
```python
classificar_texto(df, 'tratamento_5', 'sentimento')
```
Determinando a importância das palavras:
```python
from sklearn.feature_extraction.text import TfidfVectorizer


frases = ['Comprei um ótimo produto', 'Comprei um produto péssimo']

tfidf = TfidfVectorizer(lowercase=False, max_features=50)
matriz = tfidf.fit_transform(frases)
pd.DataFrame(matriz.todense(),
             columns=tfidf.get_feature_names_out())
```
```python
tfidf_bruto = tfidf.fit_transform(df["avaliacao"])
X_treino, X_teste, y_treino, y_teste = train_test_split(tfidf_bruto, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_bruto = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo: {acuracia_tfidf_bruto *100:.2f}%')
```
```python
tfidf_tratados = tfidf.fit_transform(df['tratamento_5'])
X_treino, X_teste, y_treino, y_teste = train_test_split(tfidf_tratados, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_tratados = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo: {acuracia_tfidf_tratados *100:.2f}%')
```
##**Capturando contextos:**

```python
from nltk import ngrams

frase = 'Comprei um produto ótimo'
frase_separada = token_espaco.tokenize(frase)
pares = ngrams(frase_separada, 2)
list(pares)
```
```python
tfidf_50 = TfidfVectorizer(lowercase=False, max_features=50, ngram_range=(1,2))
vetor_tfidf = tfidf_50.fit_transform(df['tratamento_5'])
X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo com 50 features e ngrams: {acuracia_tfidf_ngrams * 100:.2f}%')
```
##**Explorando a quantidade de features na vetorização*:*

```python
tfidf_100 = TfidfVectorizer(lowercase=False, max_features=100, ngram_range=(1,2))
vetor_tfidf = tfidf_100.fit_transform(df['tratamento_5'])
X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo com 100 features e ngrams: {acuracia_tfidf_ngrams * 100:.2f}%')
```
```python
tfidf_1000 = TfidfVectorizer(lowercase=False, max_features=1000, ngram_range=(1,2))
vetor_tfidf = tfidf_1000.fit_transform(df['tratamento_5'])
X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo com 1000 features e ngrams: {acuracia_tfidf_ngrams * 100:.2f}%')
```
```python
tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1,2))
vetor_tfidf = tfidf.fit_transform(df['tratamento_5'])
X_treino, X_teste, y_treino, y_teste = train_test_split(vetor_tfidf, df['sentimento'], random_state=4978)
regressao_logistica.fit(X_treino, y_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(X_teste, y_teste)
print(f'Acurácia do modelo com todas as features e ngrams: {acuracia_tfidf_ngrams * 100:.2f}%')
```
```python
vetor_tfidf.shape
```
```python
pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index=tfidf_1000.get_feature_names_out()
)
```
```python
pesos.nlargest(50, 0)
```
#**Testando o modelo de classificação**

##**Salvando e carregando o modelo**

```python
import joblib

joblib.dump(tfidf_1000, 'tfidf_vectorizer.pkl')
joblib.dump(regressao_logistica, 'modelo_regressao_logistica.pkl')
```
```python
tfidf = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')
```
```python
tfidf = joblib.load('tfidf_vectorizer.pkl')
regressao_logistica = joblib.load('modelo_regressao_logistica.pkl')
```
```python
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
token_pontuacao = tokenize.WordPunctTokenizer()
stemmer = nltk.RSLPStemmer()


def processar_avaliacao(avaliacao):
  # passo 1
  tokens = token_pontuacao.tokenize(avaliacao)

  # passo 2
  frase_processada = [palavra for palavra in tokens if palavra.lower() not in palavras_irrelevantes]

  # passo 3
  frase_processada = [palavra for palavra in frase_processada if palavra.isalpha()]

  # passo 4
  frase_processada = [unidecode.unidecode(palavra) for palavra in frase_processada]

  # passo 5
  frase_processada = [stemmer.stem(palavra) for palavra in frase_processada]

  return ' '.join(frase_processada)
```
Classificando novas avaliações:

```python
# Novas avaliações para prever
novas_avaliacoes = ["Ótimo produto, super recomendo!",
                 "A entrega atrasou muito! Estou decepcionado com a compra",
                 "Muito satisfeito com a compra. Além de ter atendido as expectativas, o preço foi ótimo",
                 "Horrível!!! O produto chegou danificado e agora estou tentando fazer a devolução.",
                 '''Rastreando o pacote, achei que não fosse recebê-lo, pois, na data prevista, estava sendo entregue em outra cidade.
                 Mas, no fim, deu tudo certo e recebi o produto.Produto de ótima qualidade, atendendo bem as minhas necessidades e por
                 um preço super em conta.Recomendo.''']
```

```python
novas_avaliacoes_processadas = [processar_avaliacao(avaliacao) for avaliacao in novas_avaliacoes]
```
```python
novas_avaliacoes_processadas
```
```python
novas_avaliacoes_tfidf = tfidf.transform(novas_avaliacoes_processadas)

predicoes = regressao_logistica.predict(novas_avaliacoes_tfidf)

df_previsoes = pd.DataFrame({
    'Avaliação': novas_avaliacoes,
    'Sentimento previsto': predicoes
})

df_previsoes
```
|index|Avaliação|Sentimento previsto|
|---|---|---|
|0|Ótimo produto, super recomendo\!|positivo|
|1|A entrega atrasou muito\! Estou decepcionado com a compra|negativo|
|2|Muito satisfeito com a compra\. Além de ter atendido as expectativas, o preço foi ótimo|positivo|
|3|Horrível\!\!\! O produto chegou danificado e agora estou tentando fazer a devolução\.|negativo|
|4|Rastreando o pacote, achei que não fosse recebê-lo, pois, na data prevista, estava sendo entregue em outra cidade\.
                 Mas, no fim, deu tudo certo e recebi o produto\.Produto de ótima qualidade, atendendo bem as minhas necessidades e por
                 um preço super em conta\.Recomendo\.|positivo|
