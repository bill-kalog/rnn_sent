## Recurrent neural networks for sentence classification

implementation of LSTM for sentiment analysis from theano [example](http://deeplearning.net/tutorial/lstm.html).
Based on this [implementation](https://github.com/inikdom/neural-sentiment)

Implemented a weighted sum version of attention based on [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) 
and [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/abs/1512.08756)

Implemetation of a combination between [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285) and [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) but only for sentiment classification. Specifically, following description as in Table 1 of [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417). 

|Module | Specification |
|-------|---------------|
|Input module|GRU|
|attention|attentionGRU|
|Mem update|ReLU|
|Mem Weights|Untied|
 

attention GRU cell architecture is based on code from [barronalex](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow)

