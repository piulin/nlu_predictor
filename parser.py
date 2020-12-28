import argparse
from argparse import RawTextHelpFormatter
from defaults import CNN_CONFIG, EPOCHS, learning_rate, hidden_size, default_pred_file


class parser(object):

    def __init__(self):
        """
        Defines the command-line arguments. Check the help [-h] to learn more.
        """
        self.parser = argparse.ArgumentParser(description='NLU predictor.', formatter_class=RawTextHelpFormatter)
        self.parser.add_argument('-b', metavar='batch_size', type=int, help='Sets the batch size.')

        self.parser.add_argument('--cuda-device', metavar='gpu_id', type=int, default=0,
                                 help='Selects the cuda device. If -1, then CPU is selected.')

        self.subparsers = self.parser.add_subparsers(title='Mode', help='Action to perform.')

        # define the test parser.
        self.test_parser = self.subparsers.add_parser('test', help='Test NLU predictor.',
                                                      formatter_class=RawTextHelpFormatter)

        self.test_parser.set_defaults(action="test")

        self.test_parser.add_argument('-m', metavar='model', type=str,
                                      help='Path to the CNN model to be loaded.', required=True)

        self.test_parser.add_argument('-test-data', metavar='dir', type=str, help='Data to be tested', required=True)
        self.test_parser.add_argument('-train-data', metavar='dir', type=str, help='To populate the indices.',
                                       required=True)

        self.test_parser.add_argument('-O', metavar='out_json', type=str, help='predicted intent and slots', default=default_pred_file)

        # define the training parser.

        self.train_parser = self.subparsers.add_parser('train', help='Train the CNN providing samples.',
                                                       formatter_class=RawTextHelpFormatter)

        self.train_parser.set_defaults(action="train")

        self.train_parser.add_argument('-C', metavar='ffnn_config', type=str,
                                       help='Loads the CNN configuration.', default=CNN_CONFIG)

        self.train_parser.add_argument('-e', metavar='epochs', type=int, default=EPOCHS,
                                       help='Configures the #epochs')

        self.train_parser.add_argument('-lr', metavar='learning_rate', type=float, default=learning_rate,
                                       help='Sets the learning rate')

        self.train_parser.add_argument('-hs', metavar='hidden_size', type=int, default=hidden_size,
                                       help='Sets the learning rate')

        self.train_parser.add_argument('-o', metavar='output_model', type=str,
                                       help='Provide a path to model to be saved.')

        self.train_parser.add_argument('-train-data', metavar='dir', type=str, help='Data used to train the network.', required=True)


    def parse_args(self):
        return vars(self.parser.parse_args())
