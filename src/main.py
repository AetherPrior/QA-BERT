"""
# Introduction
The initial Closed-Domain-QA script which we had used was not well written, and the library used (cdqa pipeline), is now deprecated.
This time, I use the Haystack libarary instead to create an Open-Domain QA system (technically, according to haystack at least)
"""


from QAmodel import QAmodel
from utils import *
from haystack.utils import print_answers
from config import *


def main():

    parser = get_parser()
    args = parser.parse_args()

    model = QAmodel(base_dir=args.base_dir,
                    exp=args.experimental,
                    gpu=args.gpu,
                    speed=args.speed,
                    reader_dir=args.reader,
                    retriever_dir=args.retriever,
                    save=args.save)

    model.build_model()

    pipline = ''

    if(args.pipeline is not None and args.pipeline not in ['read', 'retrieve']):
        print(
            f"{bcolors.FAIL}Invalid pipeline argument enter any of: read, retrieve {bcolors.ENDC}")
        return
    elif(args.pipeline is None):
        pipeline = 'default'
    else:
        pipeline = args.pipeline

    model.add_pipeline(pipeline)

    while(True):
        print("Query [or quit]: ")

        query = input()
        if(query.lower() == "quit"):
            return

        prediction = model.execute_query(query)
        print_answers(prediction)


if __name__ == '__main__':
    main()
