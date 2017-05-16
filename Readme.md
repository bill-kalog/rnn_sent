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

### hyperparameters

the following hyperparameters can be changed inside `conf.py`

|hyperparameter| description|
|--------------|------------|
|dat_directory| Directory containing datasets (used by datasets.py) |
|load_last_checkpoint| Path pointing to a directory containing tensorflow checkpoints (used only if `eval` is true) |
|eval| Choose between building/training a model (`False`) or evaluating a saved model (`True`) |
|sst_finegrained| Iff you want to use the finegrained version of [SST](https://nlp.stanford.edu/sentiment/index.html) set to `True`, otherwise `False` |
|classes_num| The number of classes of your dataset |
|bidirectional| Choose between having a bidirectional sentence encoder or not |
|GRU| Choose between using an LSTM or a GRU cell inside the encoder (pay attention though as event if set at `False` i.e LSTM some encoders (DMN) still use a GRU)|
|pooling| Choose between using avg pooling or fc layer on top of an lstm for sentence classification  |
|pool_all_output| Do avg pooling over all outputs of an RNN **BROKEN DON'T USE** |
|attention| Choose between using tensorflows' attention wrapper for a cell or not (better not use it i.e set to `False`) |
|use_attention| Use attention weights in the outputs of an RNN |
|attention_GRU| For an RNN choose between using attention weights in a weighted sum of the outputs fashion or feeding the weights to a GRU like in  [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) |
|split_dev| Choose between feeding the dev set in small batches or as a whole tensor (usefull when dev set is quite big and initializing a whole tensor would require too much memory) |
|dev_minibatch| Size of minibatches used for dev set |
|dim_proj| Dimensionality of word embeddings to be used, it is also the number of units used by a rnn cell (i.e dimensionality of hidden states/output) |
|layers| number of stacked layer to be used by an RNN encoder |
|batch_size| batch size for training set |
|n_epochs| Number of epochs over the whole dataset to be perfomrmed during training |
|n_words| Dictionary size of the training set, initialize as `None` to be caclulated in training time |
|learning_rate| learning rate of training |
|dropout_rate| dropout value for softmax layers |
|clip_threshold| Clipping threshold of RNN gradients |
|sentence_len| max sentence length to be used, initialize as `None` to be caclulated in training time  |
|keep_prob_inp| Amount of dropout during training in the input of a cell of an RNN encoder |
|keep_prob_out| Amount of dropout during training in the hidden state to be fed in the next cell of an RNN encoder  |
|l2_norm_w| Regularization term to be used on weights during optimization  |
|dmn| Choose between using a `Dynamic memory network` or a plain `RNN`|
|episodes_num| Number of episodes, if using a `DMN` |
|evaluate_every| evaluate on dev set every that many steps |
|checkpoint_every| keep a checkpoint of the model being trained every that many steps |
|save_step| save the word embeddings, to be fed, in tensorboard once after that many steps  |
|save_step_dev_info| A list containing at which steps to save some information about a model (i.e attention weights on train and dev and  dataframe for plotting in [bokeh](https://github.com/bill-kalog/bokeh_plots)) |
|std_dev| std for word embedding initialization (currently cmmented out in code though) word embeddings are initialized in the interval (-1/(2*d), 1/(2*d)) |
|train_embeddings| list of booleans stating whether trainable is going to be set `True` or `False` for the word embeddings (currently support only using one type of word embedding i.e one of the supprted or random. If using pretrained list must have length 2 with second argument set to `None`) |
|word_vector_type| list of types of pretrained word embeddings to be loaded or empty if using just random  |
|pretrained_vectors| list of paths for the pretrained vectors or and empty list if using only random |


