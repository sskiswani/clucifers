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
                    'Basic k-nearest neighbor rule.',
        usage='Specify a classifier, training data file, and testing data file.\n'
              'e.g. classfier_name path_to_training_data path_to_testing_data [-h] [-v]'
    )

    # Positional args
    parser.add_argument('clsf', type=str, default='mle',
                        help="Classifier type.\n'mle' for Maximum likelihood, 'parzen' for Parzen Windows,"
                             " and 'knn' for k-Nearest Neighbors.")
    parser.add_argument('train', type=lambda x: is_valid_file(parser, x), help='Path to training data')
    parser.add_argument('test', type=lambda x: is_valid_file(parser, x), help='Path to testing data')

    # Flag Args
    parser.add_argument('-v', '--verbose', action='store_true', help='Detailed output and debugging information')

    # Parse args
    parser.set_defaults(clsf='mle', verbose=True)

    classf.run(**vars(parser.parse_args()))
