import argparse
from argparse import RawTextHelpFormatter
from defaults import CNN_CONFIG, EPOCHS, learning_rate, hidden_size, default_pred_file, lstm_dr, lstm_bidirectional, \
    BATCH_SIZE, word_embeddings_size, slot_embeddings_size


class parser(object):

    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='NLU predictor.', formatter_class=RawTextHelpFormatter)
        self.parser.add_argument('-b', metavar='batch_size', type=int, help='Sets the batch size.',default=BATCH_SIZE)

        self.parser.add_argument('--cuda-device', metavar='gpu_id', type=int, default=0,
                                 help='Selects the cuda device. If -1, then CPU is selected.')

        self.subparsers = self.parser.add_subparsers(title='Mode', help='Action to perform.')

        # define the test parser.
        self.test_parser = self.subparsers.add_parser('test', help='Test NLU predictor.',
                                                      formatter_class=RawTextHelpFormatter)

        self.test_parser.set_defaults(action="test")

        self.test_parser.add_argument('-m', metavar='model', type=str,
                                      help='Path to the model to be loaded.', required=True)

        self.test_parser.add_argument('-test-data', metavar='dir', type=str, help='Data to be tested', required=True)
        self.test_parser.add_argument('-train-data', metavar='dir', type=str, help='To populate the indices.',
                                       required=True)

        self.test_parser.add_argument('-O', metavar='out_json', type=str, help='predicted intent and slots', default=default_pred_file)

        # continue training parser

        self.cont_parser = self.subparsers.add_parser('continue', help='continue training.',
                                                      formatter_class=RawTextHelpFormatter)

        self.cont_parser.set_defaults(action="continue")

        self.cont_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,
                                       help='Configures the #epochs')
        self.cont_parser.add_argument('-m', metavar='model', type=str,
                                      help='Path to the model to be loaded.', required=True)
        self.cont_parser.add_argument('-train-data', metavar='dir', type=str, help='Data used to train the network.',
                                       required=True)
        self.cont_parser.add_argument('-o', metavar='output_model', type=str,
                                       help='Provide a path to model to be saved.')

        self.cont_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,
                                       help='Sets the learning rate')

        self.cont_parser.add_argument('-D', metavar='test_size_percentage', type=float,
                                       help='Split the training data into train and dev. Accuracies are computed on the dev data.')

        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Learn a predictor providing samples.',
                                                       formatter_class=RawTextHelpFormatter)

        self.train_parser.set_defaults(action="train")

        self.train_parser.add_argument('-C', metavar='ffnn_config', type=str,
                                       help='Loads the FFNN configuration.', default=CNN_CONFIG)

        self.train_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,
                                       help='Configures the #epochs')

        self.train_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,
                                       help='Sets the learning rate')

        self.train_parser.add_argument('-hs', metavar='hidden_size', type=int, default=hidden_size,
                                       help='Sets the hidden state size')

        self.train_parser.add_argument('-o', metavar='output_model', type=str,
                                       help='Provide a path to model to be saved.')

        self.train_parser.add_argument('-wes', metavar='word_embeddings_size', type=int,
                                       help='Set the word embeddings size.', default=word_embeddings_size)

        self.train_parser.add_argument('-ses', metavar='slot_embeddings_size', type=int,
                                       help='Set the slot embeddings size.', default=slot_embeddings_size)

        self.train_parser.add_argument('-D', metavar='test_size_percentage', type=float, help='Split the training data into train and dev. Accuracies are computed on the dev data.')

        self.train_parser.add_argument('-train-data', metavar='dir', type=str, help='Data used to train the network.', required=True)
        self.train_parser.add_argument('-E', metavar=['file','freeze_embeddings'], nargs=2, help='Use pretrained embeddings')


        self.train_subparsers = self.train_parser.add_subparsers(title='RNN_type', help='Configure the type of seq2seq network.',required=True)
        self.rnn_parser = self.train_subparsers.add_parser('rnn', help='Use a plain seq2seq RNN.',
                                                            formatter_class=RawTextHelpFormatter)

        self.rnn_parser.set_defaults(architecture='rnn')

        self.lstm_parser = self.train_subparsers.add_parser('lstm', help='Use a seq2seq LSTM.',
                                                      formatter_class=RawTextHelpFormatter)

        self.lstm_parser.set_defaults(architecture='lstm')

        self.lstm_parser.add_argument('-bidirectional',
                                      help='Decide to use a bidirectional LSTM or not.', action='store_true', default=lstm_bidirectional)

        self.lstm_parser.add_argument( '-dropout', metavar='dropout', type=float,
                                      help='Set the dropout of the LSTM.', default=lstm_dr )


    def parse_args(self):
        return vars(self.parser.parse_args())
