import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar a tabela de emoções
tabela_emocoes = "emotions.csv"
# colocar em um DataFrame, que é uma tabela inteligente
df = pd.read_csv(tabela_emocoes)

# Mostrar as 5 primeiras linhas da tabela
print("--- Amostra dos Dados da Tabela de Emoções ---")
print(df.head())

# Mostrar informações sobre as colunas da tabela
print("\n--- Informações do DataSet ---")
df.info()

# Mostrar estatísticas descritivas (média, desvio padrão, etc.)
print("\n--- Estatísticas Descritivas ---")
print(df.describe())

"""
O DataSet possui:
1. 2132 linhas (0 a 2131)
2. 2549 colunas
    2.1. Sendo 2548 do tipo float64 e um objeto do tipo label
3.  41.5 MB de memória
"""

"""
Explicação das Colunas:
1. [tipo_de_feature]_[numero_da_feature]_[banda_cerebral]
2. Prefixo: indica o tipo de features que foi calculada. Qual operação matemática foi aplicada. Exemplo:
    - mean_0_a:
        + mean: média aritmética
        + 0: número da feature
        + a: banda alfa
"""

# X recebe todas as colunas, exceto a última
x = df. drop('label', axis=1)

# Y recebe apenas a coluna "label"
y = df['label']

print("\n--- Dados Separados ---")
print("Formato das Features (X): ", x.shape)
print("Formato do Rótulo (Y): ", y.shape)

"""
Explicação do Retorno no Console desta Etapa:
1. A variável X contém:
    - 2132 linhas (cada linha é uma amostra)
    - 2548 colunas (cada coluna é uma feature que descreve a amostra)
2. A variável Y contém:
    - 2132 elementos (um para cada linha da matriz X)
    - cada elemento é uma label dos elementos da matriz X
"""

# random_state=42 garante que a divisão seja sempre a mesma, para reprodutibilidade
# stratify=y garante que a proporção de emoções seja a mesma no treino e no teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
"""
Explicação dos Parãmetros:
    1. test_size=0.2: 20% dos dados serão usados para teste, 80% para o treinamento
    2. random_state=42: parâmetro de reprodutibilidade. Antes de dividir os dados, ele embaralha os dados aleatoriamente para que o treino
        contenha uma mistura de exemplos. Garantindo que o embaralhamento seja o mesmo em execuções diferentes.
    3. stratify=y: mantém a mesma proporção de classes (cada emoção) no conjunto de treino e no conjunto de teste.
Explicação do Retorno:
    1. x_train: são 1705 amostras (80% do total de 2132) com as mesmas 2548 features.
    2. y_train: são as respostas para os 1705 exemplos de treino.
    3. x_test: são as 427 amostras restantes (20% do restante total) com as 2548 features. 
    4. y_test: são para calcular a acurácia final.
"""
print("\n--- Dados Divididos ---")
print("Tamanho do conjunto de treino:", x_train.shape)
print("Tamanho do conjunto de teste:", x_test.shape)

# Cria o modelo com 100 árvores de decisão
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Treina o modelo com os dados de treino
print("\n--- Treinando o Modelo ---")
modelo.fit(x_train, y_train)
print("\n--- Modelo Treinnado com Sucesso ---")

# 1. Usa o modelo treinado para prever as emoções nos dados de teste
print("\n --- Fzendo Previsões ---")
previsoes = modelo.predict(x_test)

# 2. Compara as predições com os rótulos verdadeiros
acuracia = accuracy_score(y_test, previsoes)
print(f"\nAcurácia do Modelo: {acuracia * 100:.2f}%")

# 3. Gera um relatório mais detalhado (precisão, recall, f1-score por emoção)
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, previsoes))

# 4. Gera a matriz de Confusão para ver onde o modelo mais erra
print("\n--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

"""
Explicação dos Resultados: 
    1. Acurácia do Modelo: de todas as 427 amostras do conjunto de teste,, ele acertou a emoção corrta em 98.83% das vezes.
    2. Relatório de Classificação: este relatório detalha a performance para cada uma das classes
        2.1 precision: mostra a precisão de acertos dos labels previstos para cada emoção.
        2.2 recall: mostra a capacidade do modelo de encontrar todas as instâncias de cada emoção.
        2.3 f1-score: é a média harmônica entre precisão e recall, uma métrica para avaliar o equilíbrio do modelo.
        2.4 support: contagem de quantas amostras de cada classe existiam no conjunto de teste.
    3. Matriz de Confusão: mostra exatamente como foram os acertos e os erros. As linhas são os valores e as colunas são as previsões.
        A ordem é [NEGATIVE, NEUTRAL, POSITIVE];
        [[140   0   2]
        [  0 143   0]
        [  3   0 139]]
        - 140:140 amostras que eram NEGATIVE foram corretamente previstas como NEGATIVE.
        - 143:143 amostras que eram NEUTRAL foram corretamente classificadas como NEUTRAL.
        - 139: 139 amostras que eram POSITIVE foram corretamente classificadas como POSITIVE
        - houve 2 amostras que eram NEGATIVE e foram confundidas como POSITIVE.
        - houve 3 amostras que eram POSITIVE e foram confundidas como NEGATIVE.
"""