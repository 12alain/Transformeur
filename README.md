


1. [Transformeurs](#Transformeurs)

    1.1 [Définition](#définition)

    1.2 [Pipeline des Transformeurs](#pipeline-des-transformeurs)

    1.3 [Comparaison avec les modèles précédents (RNN, LSTM)](#comparaison-avec-les-modèles-précédents-rnn-lstm)

2. [Architecture des Transformeurs](#architecture-des-transformeurs)

    2.1 [Input](#input)

    2.2 [Tokenisation](#tokenisation)

    2.3 [Embeddings](#embeddings)

    2.4 [Position encoding](#position-encoding)

    2.5 [Self-attention](#self-attention)

![Logo](https://www.researchgate.net/publication/323904682/figure/fig1/AS:606458626465792@1521602412057/The-Transformer-model-architecture.png)
# Transformeurs

Bien sûr, je peux vous aider à élaborer un plan détaillé sur les transformeurs, leur utilisation dans les pipelines, les encodeurs et les décodeurs, ainsi que leur application dans la traduction de langue. Voici un plan structuré :

# definition

ien sûr, je peux vous aider à élaborer un plan détaillé sur les transformeurs, leur utilisation dans les pipelines, les encodeurs et les décodeurs, ainsi que leur application dans la traduction de langue. Voici un plan structuré :

#pipeline-des-transformeurs

ien sûr, je peux vous aider à élaborer un plan détaillé sur les transformeurs, leur utilisation dans les pipelines, les encodeurs et les décodeurs, ainsi que leur application dans la traduction de langue. Voici un plan structuré :



## Features

- Light/dark mode toggle
- Live previews
- Fullscreen mode
- Cross platform


![Logo](https://www.researchgate.net/publication/323904682/figure/fig1/AS:606458626465792@1521602412057/The-Transformer-model-architecture.png)


## Deployment

To deploy this project run

```bash
  npm run deploy
```


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Support

For support, email fake@fake.com or join our Slack channel.


## Usage/Examples

```javascript
class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
            super().__init__()
            self.d_model=d_model
            self.vocab_size=vocab_size
            self.embedding=nn.Embedding(vocab_size,d_model)
            print(self.embedding)
    def forward(self , x):
        
        return self.embedding(x)*math.sqrt(self.d_model)
}
```



