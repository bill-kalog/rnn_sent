# configuration file
config = {

    'dat_directory': '../datasets',
    'sst_finegrained': False,  # used only when/if loading SST choooses [5, 2] classes
    'classes_num': 2,  # number of classes

    'bidirectional': True,  # bidirectional rnn or not
    'pooling': True,  # choose between pooling or fc layer
    'dim_proj': 300,  # word embeding dimension and LSTM number of hidden units.
    'layers': 1,
    'batch_size': 120,  # The batch size during training.
    'n_epochs': 50,
    'n_words': None,  # Leave as None, dictionary size
    'learning_rate': 1e-4,
    'dropout_rate': 0.7,
    'clip_threshold': 6,
    'sentence_len': None,  # max sentence length
    'keep_prob_inp': 1.0,
    'keep_prob_out': 1.0,

    'evaluate_every': 5,  # evaluate on dev set

    'save_step': 500,

    # word embeddings args
    'std_dev': 0.01,  # variance
    'train_embeddings': [True, None],
    'word_vector_type': ['glove'],
    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.100d.txt'],

    'train_embeddings': [True, None],
    'word_vector_type': ['glove'],
    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt'],

    # 'train_embeddings': [True],
    # 'word_vector_type': [],
    # 'pretrained_vectors': [],














    # 'patience': 10,  # Number of epoch to wait before early stop if no progress
    # 'max_epochs': 5000,  # The maximum number of epoch to run
    # 'dispFreq': 10,  # Display to stdout the training progress every N updates
    # # 'decay_c': 0.,  # Weight decay for the classifier applied to the U weights.
    # # 'lrate': 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # # 'n_words': 10000,  # Vocabulary size
    # # 'optimizer': adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    # # 'encoder': 'lstm',  # TODO: can be removed must be lstm.
    # # 'saveto': 'lstm_model.npz',  # The best model will be saved there
    # 'validFreq': 370,  # Compute the validation error after this number of update.
    # 'saveFreq': 1110,  # Save the parameters after every saveFreq updates
    # 'maxlen': 100,  # Sequence longer then this get ignored

    # 'valid_batch_size': 64,  # The batch size used for validation/test set.


    # #  # # dataset to load arguments
    # # 'dat_directory': '../datasets',  # parent directory where a dataset is stored
    # # 'sst_finegrained': True,  # used only when/if loading SST choose [5, 2] classes
    # # 'classes_num': 5,  # number of classes


    # # # # network architecture specific arguments
    
    # 'seperate_filters': False,  # choose between seperate filters per word \
    # # embedding matrix or not (True, False)
    # 'clipping_weights': False,
    # 'save_step': 1000,  # step at which word embedings will be saved
    # # 'kernel_sizes': [7, 7, 7, 7],
    # 'kernel_sizes': [3, 4, 5],
    # # 'kernel_sizes': [3, 4, 5, 6, 7, 7],
    # # 'kernel_sizes': [7, 7, 7, 8, 8, 8],
    # # decide which embeddings to finetune (must have a value for the random
    # # vector (last one) as well. specifically for the random vector can be
    # # True, False or None (if want to skip completely)
    # # 'train_embeddings': [False, True, True],  # True],
    # # 'train_embeddings': [False, True, True, True, True, None],
    # # 'train_embeddings': [False, True, True, None],
    # # 'train_embeddings': [False, True, None],  # , True]
    # 'train_embeddings': [False, True, None],
    # # 'train_embeddings': [True],
    
    # # 'learning_rate': 1e-3,
    # # 'learning_rate': 1e-4,
    # # 'learning_rate': 1e-5,
    # 'edim': 300,  # dimension of word embeddings

    # # 'std_dev': 0.05,
    # 'std_dev': 0.01,  # variance

    # 'n_filters': 100,  # number of filters per kernel
    # # 'batch_size': 128,
    # 'l2_regularization': 3,  # weight of l2 regularizer
    # 'evaluate_every': 5,  # evaluate on dev set
    # 'checkpoint_every': 200,  # strore a checkpoint
    # 'num_checkpoints': 5,
    # # 'paths': ['data/rt-polarity.pos', 'data/rt-polarity.neg']
    # # 'pretrained_vectors': '../datasets/glove_6B/glove.6B.100d.txt'


    # # # # pretrained networks arguments
    # # type of each pretrained word vector to be loaded (based on
    # # values available/implemented inside word_vectors classs)
    # # leave these two lists empty when not using pretrained embeddings
    # 'word_vector_type': ['glove', 'glove'],
    # # 'word_vector_type': ['fastText', 'fastText', 'glove'],
    # # 'word_vector_type': ['glove', 'glove'],
    # # 'word_vector_type': ['levy', 'levy'],
    # # 'word_vector_type': ['fastText', 'fastText'],
    # # 'word_vector_type': ["W2V", "W2V"],
    # # 'word_vector_type': ['glove', 'glove', 'fastText', 'W2V', 'levy'],
    # # 'word_vector_type': ["W2V"],
    # # 'word_vector_type': ['glove'],
    # # 'word_vector_type': [],  # use only random vectors

    # # 'pretrained_vectors': ['../datasets/glove_42B/glove.42B.300d.txt']
    # 'pretrained_vectors': ['../datasets/fastText/wiki.en.vec',
    #                        '../datasets/fastText/wiki.en.vec',
    #                        '../datasets/glove_6B/glove.6B.300d.txt'],

    # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt',
    #                        '../datasets/glove_6B/glove.6B.300d.txt'],

    # # 'pretrained_vectors': ['../datasets/fastText/wiki.en.vec',
    # #                        '../datasets/fastText/wiki.en.vec'],

    # # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.100d.txt',
    # #                        '../datasets/glove_6B/glove.6B.100d.txt'],

    # # 'pretrained_vectors': [
    # #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin',
    # #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin'],

    # # 'pretrained_vectors': ['../datasets/levy/bow5.words',
    # #                        '../datasets/levy/bow5.words'],

    # # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt',
    # #                        '../datasets/glove_6B/glove.6B.300d.txt',
    # #                        '../datasets/fastText/wiki.en.vec',
    # #                        '../datasets/w2vec/GoogleNews-vectors-negative300' +
    # #                        '.bin',
    # #                        '../datasets/levy/bow5.words'],

    # # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt'],
    # # 'pretrained_vectors': [
    # #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin'],

    # # 'pretrained_vectors': []  # use only random vectors

}
