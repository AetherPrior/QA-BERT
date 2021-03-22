"""
# Introduction
The initial Closed-Domain-QA script which we had used was not well written, and the library used (cdqa pipeline), is now deprecated. 
This time, I use the Haystack libarary instead to create an Open-Domain QA system (technically, at least)
"""


"""
# Method

- I have used a Dense-passage-retrieval System as a retriever for QA, and and existing BERT Models that function as document embedders and query embedders

- We did *not* initially train the model at all, but I'm researching into ways to be able to train it on several similar datasets to improve the performance. 
    - For example, there are ways to train the DPR using a medical QA dataset. I'm trying to find one that's more relevant to our field.

- Colab is NOT the way to go IMO for any decent code, but I'm constrained by the lack of a GPU on my system for speed. 
    - The initial code was written on colab, after which I've exported it
    - Python code can actually be run on colab by mounting it to the drive and calling it from colab


"""




from QAmodel import QAmodel
import argparse
import os
def path_dir(path):

    if(os.path.isdir(path)):
        return path
    else:
        raise "Not a directory"


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
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    model = QAmodel(base_dir=args.base_dir,
                    exp=args.experimental,
                    gpu=args.gpu,
                    speed=args.speed)

    model.build_model()

    pipline = ''
    '''
    if(args.pipeline is not None and parser.pipeline not in ['read', 'retrieve']):
        print("Invalid pipeline argument enter any of: read, retrieve")
        return
    elif(args.pipeline is None):
        pipeline = 'default'
    else:
        pipeline = args.pipeline

    model.add_pipeline(pipeline)

    while(True):
        print("Query: ")
        query = input()
        prediction = model.execute_query(query)

    '''


if __name__ == '__main__':
    main()
