from config import *


class QAmodel:
    def __init__(self, mode, docs):
        self._docs = docs
        self.model = None
        self._converter = None
        self._doc_store = None
        self._processor = None
        self._retriever = None
        self._reader = None
        self._mode = mode
        self._pipeline = None

    def execute_query(self, query):
        '''
        We attempt to find the answer to a query from the retriever based the model
        prerequisite: The model is built
        postcondition: The answer(s) is(are) generated and returned
        '''

        if (self.model is None):
            print(
                "A model has not been built! We can build it if you wish: [y/N]")
            opt = input()
            if (opt.lower() == 'y'):
                self.build_model()
            else:
                return

        prediction = pipe.run(query="What is a parser?",
                              top_k_retriever=5, top_k_reader=5)

    def build_model(self):

        # Experimental vs Actual
        if (self._mode & EXP == 1):
            self._doc_store = InMemoryDocumentStore(similarity="dot_product")
        else:
            self._doc_store = FAISSDocumentStore(similarity="dot_product")

        self._processor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by='passage',
            split_overlap=10,
            split_respect_sentence_boundary=False
        )

        doc_p = processor.process(doc)
        self._doc_store.write_documents(doc_p)

        self._retriever = DensePassageRetriever(
            document_store=doc_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
        )

        self._doc_store.update_embeddings(retriever)

        # for poor guys like me who don't have a gpu
        gpu_yn = True if self._mode & GPU == 1 else False

        if (self._mode & SPEED == 1):
            # prefer speed over acc, people are impatient
            self._reader = FARMReader(
                model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=gpu_yn)
        else:
            # prefer acc over spped, someone who likes to wait
            self._reader = FARMReader(
                model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=gpu_yn)

    def add_pipeline(self, pipeline='default'):
        if (pipeline == 'default'):
            self._pipeline = ExtractiveQAPipeline(
                self._reader, self._retriever)
