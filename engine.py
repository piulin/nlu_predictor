import sys

import gensim
import torch

from ai.seq2se2q_rnn import seq2seq
from assessment import assess
from dataset import dataset, test_dataset
from writer import predictions2json


def get_device(args):
    # get our working device
    return torch.device("cuda:" + str(args['cuda_device'])
                          if torch.cuda.is_available() and args['cuda_device'] != -1
                          else "cpu")

def train(args):
    """
    Train the CNN.
    :param args:
    """

    # get our working device
    device = get_device(args)

    # read training samples from disk, and prepare the dataset (i.e., shuffle, scaling, and moving the data to tensors.)
    dts = dataset(device)

    dts.read_training_dataset( args ['train_data'] )

    train_dts, test_dts = dts, dts

    if args['D'] != None:
        train_dts, test_dts = dts.train_test_splits(args['D'], td=False)



    classifier = seq2seq(dts.words_converter.no_entries(),
                          dts.slots_converter.no_entries(),
                          dts.intent_converter.no_entries(),
                          device,
                         train_dts.words_converter.T2id('<PAD>'),
                         train_dts.slots_converter.T2id('<PAD>'),
                          args)

    if args['E'] != None:
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(args['E'][0],
                                                                     binary=True)
        classifier.pretrained_embeddings( train_dts, embeddings )


    try:
        classifier.fit( train_dts, test_dts, args ['e'], args['b'],  args['lr'], args['save_every'] ,
                        args ['o'], do_predict=not args['no_training_predictions'])
    except KeyboardInterrupt:
        pass


    intent_true, intent_pred, slots_true, slots_pred = classifier.predict_and_get_labels_batch(test_dts, args['b'])
    assess(dts,intent_true, intent_pred, slots_true, slots_pred)

    if args['o'] != None:

        # write it into a file.
        classifier.dump( args ['o'] )



def cont(args):
    # get our working device
    device = get_device(args)

    dts = dataset(device)

    dts.read_training_dataset(args['train_data'])

    train_dts, test_dts = dts, dts

    if args['D'] != None:
        train_dts, test_dts = dts.train_test_splits(args['D'])

    classifier = seq2seq.load( args ['m'] )

    try:
        classifier.fit(train_dts, test_dts, args ['e'], args['b'],  args['lr'] )
    except KeyboardInterrupt:
        pass

    if args['o'] != None:
        # write it into a file.
        classifier.dump(args['o'])





def test (args):
    """
    Predict on a test dataset given a model.
    :param args:
    """

    device = get_device(args)

    net = seq2seq.load( args['m'] )

    dts = dataset(device)
    dts.read_training_dataset(args['train_data'])


    test  = test_dataset(device,
                    words_converter=dts.words_converter,
                    slots_converter=dts.slots_converter,
                    intent_converter=dts.intent_converter)

    if args['E'] != None:
        test.read_test_dataset( args['test_data'], lock=False )
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(args['E'],
                                                                     binary=True)
        net.pretrained_embeddings(test, embeddings)

    else:
        test.read_test_dataset( args['test_data'], lock=True )

    print(dts.intent_converter.no_entries())
    # # predict!
    intent_pred, slots_pred = net.predict_batch( test, args['b'] )


    predictions2json(test, intent_pred, slots_pred, args['O'])

