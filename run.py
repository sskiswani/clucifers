import argparse
import os.path
import classf


def is_valid_file(argparser, arg):
    """
    ref: https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va

    :param argparser:
    :param arg:
    :return:
    """
    fpath = os.path.abspath(arg)
    if not os.path.exists(fpath):
        argparser.error("Could not find the file %s." % arg)
    else:
        return fpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='CAP5638 Project 1: Classification Using Maximum-likelihood, Parzen Window, and K-Nearest Neighbor',
        description='Implementation of Maximum-likelihood and and Parzen Window Bayesian classifers and '
                    'Basic k-nearest neighbor rule.'
    )

    # Positional args
    parser.add_argument('classifier_name', type=str, default='mle',
                        help="Classifier type. 'mle' for Maximum likelihood, 'parzen' for Parzen Windows,"
                             " and 'knn' for k-Nearest Neighbors.")
    parser.add_argument('training_data', type=lambda x: is_valid_file(parser, x), help='Path to training data')

    # Flag Args
    parser.add_argument('-v', '--verbose', action='store_true', help='Detailed output and debugging information')
    parser.add_argument('-t', '--testing_data', type=lambda x: is_valid_file(parser, x),
                        help='Path to data used for testing the classifier.')
    parser.add_argument('-c', '--classify_data', type=lambda x: is_valid_file(parser, x),
                        required=False,
                        help='Path to data used for classification.')

    # Parse args
    parser.set_defaults(verbose=True, classify_data=None, testing_data=None)

    classf.run(**vars(parser.parse_args()))
