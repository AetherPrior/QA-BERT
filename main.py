"""
# Introduction
The initial Closed-Domain-QA script which we had used was not well written, and the library used (cdqa pipeline), is now deprecated. 
This time, I use the Haystack libarary instead to create an Open-Domain QA system (technically, at least)
"""


"""
# Method

- I have used a Dense-passage-retrieval System as a retriever for QA, and and existing BERT Models that function as document embedders and query embedders

- We did *not* initially train the model at all, but I'm researching into ways to be able to train it on several similar datasets to improve the performance. 
    - For example, there are ways to train the [DPR](https://haystack.deepset.ai/docs/latest/tutorial9md) using [medical QA datasets](https://github.com/abachaa/Existing-Medical-QA-Datasets). 
        
    -I'm trying to find one that's more relevant to our field.

- Colab is NOT the way to go IMO for any decent code, but I'm constrained by the lack of a GPU on my system for speed. 
    - The initial code was written in colab, after which I've exported it, and changed it
    - Python code can actually be run on colab by mounting it to the drive and calling it from colab


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
