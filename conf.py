# configuration file
config = {

    'dat_directory': '../datasets',
    # "load_last_checkpoint": "./runs/1496428547/checkpoints",
    "load_last_checkpoint": "./runs/1497313304/best_snaps",
    'eval': False,  # train or evaluate model

    'sst_finegrained': True,  # used only when/if loading SST choooses [5, 2] classes
    'classes_num': 5,  # number of classes

    'bidirectional': True,  # bidirectional rnn or not
    'GRU': True,  # chose between GRU or LSTM, seems to work

    # RNN speific
    'pooling': False,  # choose between avg pooling or fc layer in the end
    'pool_all_output': False,  # Do avg pooling over all outputs of RNN ** BROKEN DOESN'T USE **
    'attention': False,  # choose between tensorflows' attention or not (better not use it)
    'use_attention': True,  # use attention on the rnn outputs
    'attention_GRU': False,  # for the simple RNN choose between producing the sentence representation through a weighted sum of attention or through an attention GRU as presented in  `Dynamic Memory Networks for Visual and Textual Question Answering`
    'split_dev': True,  # calculate dev set metrics in minibatches
    'dev_minibatch': 100,  # minibatch used for dev set if dev set tensor too big to fit in memory (looking at you bidirectional networks)
    'dim_proj': 300,  # both word embeding dimension and RNN number of hidden units.
    'layers': 1,
    'batch_size': 120,  # The batch size during training.
    'n_epochs': 200,
    'n_words': None,  # Leave as None, dictionary size
    'learning_rate': 1e-4,
    'dropout_rate': 0.5,
    'clip_threshold': 6,
    'sentence_len': None,  # max sentence length, leave as None
    'keep_prob_inp': .5,
    'keep_prob_out': .5,
    # 'keep_prob_inp': 1.,
    # 'keep_prob_out': 1.,
    'l2_norm_w': 0.00,

    # DMN specific
    'dmn': False,  # use a dynamic memory network
    'episodes_num': 1,  # number of episodes in episodic module



    # train procedure specific
    'evaluate_every': 1,  # evaluate on dev set
    'checkpoint_every': 500,  # keep a model checkpoint every 50 steps

    'save_step': 800,  # save word embeddings
    # 'save_step_dev_info': [2, 50, 100, 500, 800, 1200, 1500, 2000, 4000],
    'save_step_dev_info': [50, 100, 500, 800, 1000, 1200, 1500, 1700, 2000, 2500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000],

    # 'save_step_dev_info': [1500, 2000, 4000],

    # word embeddings args
    'std_dev': 0.01,  # variance
    # 'std_dev': 0.08,  # variance
    # 'train_embeddings': [True, None],
    # 'word_vector_type': ['glove'],
    # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.100d.txt'],

    'train_embeddings': [True, None],
    'word_vector_type': ['glove'],
    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt'],

    # 'train_embeddings': [True, None],
    # 'word_vector_type': ['W2V'],
    # 'pretrained_vectors': ['../datasets/w2vec/GoogleNews-vectors-negative300' +
    #                        '.bin'],

    # random vectors only
    # 'train_embeddings': [True],
    # 'word_vector_type': [],
    # 'pretrained_vectors': [],
    # 'word_vector_type': ['glove', 'fastText', 'W2V', 'levy'],
    # 'pretrained_vectors': [
    #                        '../datasets/glove_6B/glove.6B.300d.txt',
    #                        '../datasets/fastText/wiki.en.vec',
    #                        '../datasets/w2vec/GoogleNews-vectors-negative300' +
    #                        '.bin',
    #                        '../datasets/levy/bow5.words'],

}
