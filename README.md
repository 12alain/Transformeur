

1. [Transformeurs](#Transformeurs)

    1.1 [La Fonction pipeline ](#La-fonction-pipeline)

    1.2 [Comparaison avec les modèles précédents (RNN, LSTM)](#comparaison-avec-les-modèles-précédents-rnn-lstm)

2. [Architecture des Transformeurs](#architecture-des-transformeurs)

    2.1 [Input](#input)

    2.2 [Tokenisation](#tokenisation)

    2.3 [Embeddings](#embeddings)

    2.4 [Position encoding](#position-encoding)

    2.5 [Self-attention](#self-attention)

![Logo](https://www.researchgate.net/publication/323904682/figure/fig1/AS:606458626465792@1521602412057/The-Transformer-model-architecture.png)




# Transformeurs

Les transformeurs sont une architecture de réseau neuronal utilisée principalement dans le traitement du langage naturel (NLP) et dans les tâches de traduction automatique. Ils ont été introduits dans le papier "Attention is All You Need" par Vaswani et al. en 2017. 


# La fonction pipeline

L'objet le plus fondamental dans la bibliothèque 🤗 Transformers est la fonction `pipeline()`. Elle connecte un modèle avec ses étapes de prétraitement et de post-traitement nécessaires, nous permettant d'entrer directement n'importe quel texte et d'obtenir une réponse intelligible.

```javascript
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

```javascript
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```



# Comparaison avec les modèles précédents (RNN, LSTM)


| Caractéristique                                | Transformeurs                                 | RNN                                       | LSTM                                      |
|------------------------------------------------|-----------------------------------------------|-------------------------------------------|-------------------------------------------|
| **Architecture**                               | Basé sur des mécanismes d'attention           | Réseau de neurones récurrent              | RNN avec des cellules de mémoire          |
| **Parallélisme**                               | Oui (traitement parallèle des séquences)      | Non (traitement séquentiel)               | Non (traitement séquentiel)               |
| **Complexité de calcul**                       | Moins efficace pour les séquences très longues | Efficace pour les séquences courtes       | Efficace pour les séquences moyennes      |
| **Performance**                                | State-of-the-art sur de nombreuses tâches NLP | Bon pour les tâches séquentielles simples | Bon pour les tâches séquentielles complexes |
| **Entraînement**                               | Nécessite plus de ressources (mémoire et calcul) | Moins coûteux                             | Coût modéré                               |
| **Applications typiques**                      | Traduction, génération de texte, résumé, etc. | Prévision de séries temporelles, génération de séquences simples | Prévision de séries temporelles complexes, génération de séquences plus complexes |

