# Análise e Classificação de Emoções com EEG e Machine Learning

Este projeto é um estudo prático que implementa um pipeline completo de Machine Learning para classificar emoções (Positivo, Negativo, Neutro) a partir de um dataset de sinais de EEG pré-processados.

Este trabalho representa um marco inicial nos estudos para o meu TCC em Ciência da Computação, focado na área de Computação Afetiva e Interfaces Cérebro-Computador.

---

### Dataset

O dataset utilizado foi o **"EEG Brainwave Dataset: Feeling Emotions"**, disponível publicamente na plataforma Kaggle.

- **Link para o Dataset:** [https://www.kaggle.com/datasets/shashwatwork/eeg-brainwave-dataset-feeling-emotions](https://www.kaggle.com/datasets/shashwatwork/eeg-brainwave-dataset-feeling-emotions)
- **Importante:** Este dataset já contém características extraídas do sinal de EEG nos domínios do tempo e da frequência, permitindo focar diretamente na etapa de modelagem.

---

### Metodologia

O pipeline implementado neste script segue os seguintes passos:
1.  **Análise Exploratória dos Dados (EDA):** Carregamento dos dados com Pandas e inspeção inicial da estrutura.
2.  **Pré-processamento para ML:** Separação dos dados em features (X) e rótulos (y).
3.  **Divisão dos Dados:** Separação em conjuntos de treino (80%) e teste (20%) de forma estratificada.
4.  **Treinamento do Modelo:** Utilização do classificador `RandomForestClassifier` do Scikit-learn.
5.  **Avaliação:** Análise de performance do modelo no conjunto de teste com métricas como Acurácia, Relatório de Classificação e Matriz de Confusão.

---

### Resultados

O modelo treinado alcançou uma acurácia de **98.83%** no conjunto de teste, demonstrando alta capacidade de generalização e eficácia das features para a tarefa de classificação.

---

### Como Executar

1. Clone este repositório: `git clone [URL_DO_SEU_REPO]`
2. Instale as dependências: `pip install -r requirements.txt`
3. Baixe o dataset do link do Kaggle acima e coloque o arquivo `emotions.csv` na mesma pasta do script.
4. Execute o script Python: `python analise_emocoes.py`