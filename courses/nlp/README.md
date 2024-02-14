# Natural Language Processing - NLP

- Recurrent neural networks
- Sentiment analysis
- Character generation

## Textual data -> numeric data

### Bag of Words

- Cria um vocubulário de palavras para a entreda, atribuindo um número para cada palavra.
- Ocorre a perda de contexto da palavra, salvando apenas a frequência das palavras.

### Word Embeddings

- Representa cada palavra como um vetor.
- É uma camada do modelo, sendo necessário ser aprendido ao longo do modelo.

### Recurrent Neural Networks

- Possui loops internos para que seja refeito o processamento sobre uma palavra passada por ela anteriormente.

#### LSTM

- Salva informações processadas anteriormente para as camadas mais adiante. Serve como uma pequena memória.

