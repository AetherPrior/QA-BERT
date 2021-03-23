import argparse
import os


def get_parser():
    '''
    Helper function for argument parsing
    Returns a parser
    '''

    parser = argparse.ArgumentParser(
        description='A program to assist with question answering')

    required = parser.add_argument_group('required args')

    required.add_argument(
        '--text-dir',
        help='Path to textbook directory for searching',
        type=path_dir,
        action='store',
        dest='base_dir',
        required=True
    )

    parser.add_argument(
        '--gpu',
        help='Utilizes the GPU',
        action='store_true',
        dest='gpu'
    )

    parser.add_argument(
        '--experimental',
        help='Creates a model for experimental use',
        action='store_true',
        dest='experimental'
    )

    parser.add_argument(
        '--speed',
        help='Chooses a model to tradeoff accuracy for speed',
        action='store_true',
        dest='speed'
    )

    # generator tbd
    parser.add_argument(
        '--pipeline',
        help='''
              Accepts either 'retrieve' or 'read' (default = 'read')
              Chooses between a document retriever or a reader+retriever.  
              To see what to choose check: https://haystack.deepset.ai/docs/latest/tutorial1md and the following pages
              
              ''',
        action='store',
        dest='pipeline'
    )

    parser.add_argument(
        '--retriever',
        help="Specify saved retriever dir",
        action='store',
        dest='retriever'
    )

    parser.add_argument(
        '--reader',
        help="Specify saved reader dir",
        action='store',
        dest='reader'
    )

    parser.add_argument(
        '--no-save',
        help="Don't save the model",
        action='store_false',
        dest='save'
    )
    return parser


def path_dir(path):
    '''
    Checks if a directory is valid
    '''
    if(os.path.isdir(path)):
        return path
    else:
        raise NotADirectoryError()
