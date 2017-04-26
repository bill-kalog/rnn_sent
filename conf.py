# configuration file
config = {

    'dat_directory': '../datasets',
    'sst_finegrained': False,  # used only when/if loading SST choooses [5, 2] classes
    'classes_num': 2,  # number of classes
    'dmn': False,  # use a dynamic memory network

    'bidirectional': False,  # bidirectional rnn or not
    'GRU': True,  # chose between GRU or LSTM, seems to work
    'pooling': False,  # choose between avg pooling or fc layer in the end
    'pool_all_output': False,  # Do avg pooling over all outputs of RNN ** BROKEN DOESN'T USE **
    'attention': False,  # choose between tensorflows' attention or not (better not use it)
    'use_attention': False,  # use attention on the rnn outputs
    'split_dev': True,  # calculate dev set metrics in minibatches
    'dev_minibatch': 100,  # minibatch used for dev set if dev set tensor too big to fit in memory (looking at you bidirectional networks)
    'dim_proj': 300,  # word embeding dimension and LSTM number of hidden units.
    'layers': 1,
    'batch_size': 120,  # The batch size during training.
    'n_epochs': 200,
    'n_words': None,  # Leave as None, dictionary size
    'learning_rate': 1e-4,
    'dropout_rate': 0.7,
    'clip_threshold': 6,
    'sentence_len': None,  # max sentence length, leave as None
    'keep_prob_inp': .5,
    'keep_prob_out': .5,
    'keep_prob_inp': 1.,
    'keep_prob_out': 1.,

    'evaluate_every': 5,  # evaluate on dev set

    'save_step': 800,
    'save_step_dev_info': 800,

    # word embeddings args
    'std_dev': 0.01,  # variance
    'train_embeddings': [True, None],
    'word_vector_type': ['glove'],
    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.100d.txt'],

    'train_embeddings': [True, None],
    'word_vector_type': ['glove'],
    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt'],

    # random vectors only
    'train_embeddings': [True],
    'word_vector_type': [],
    'pretrained_vectors': [],

}
