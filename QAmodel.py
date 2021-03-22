from config import *
import os

# File conversion
from haystack.file_converter.pdf import PDFToTextConverter

# Document Stores
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore

# Preprocessing from haystack's side
from haystack.preprocessor.preprocessor import PreProcessor

# Dense passage retrieval for answering
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader

from haystack.utils import print_answers

# Different pipelines for each model
from haystack.pipeline import ExtractiveQAPipeline, DocumentSearchPipeline, GenerativeQAPipeline


class QAmodel:
    def __init__(self, base_dir, exp=False, speed=True, gpu=False):
        # data
        self._base_dir = base_dir

        # model (built?)
        self._model = False

        # components of the pipeline
        self._converter = None
        self._doc_store = None
        self._processor = None
        self._retriever = None
        self._reader = None

        # modes
        self._exp, self._speed, self._gpu = exp, speed, gpu
        self._pipeline = None

    def execute_query(self, query):
        '''
        We attempt to find the answer to a query from the retriever based the model
        prerequisite: The model is built
        postcondition: The answer(s) is(are) generated and returned
        '''

        if (not self._model):
            print(
                "A model has not been built! We can build it if you wish: [y/N]")
            opt = input()
            if (opt.lower() == 'y'):
                self.build_model()
                self.add_pipeline()
            else:
                return

        elif(self._pipeline is None):
            print(
                "A pipeline has not been attached! We can build it if you wish: [y/N]")
            opt = input()
            if (opt.lower() == 'y'):
                self.build_model()
                self.add_pipeline()
            else:
                return

        prediction = self._pipeline.run(query=query,
                                        top_k_retriever=5, top_k_reader=5)

        return prediction['answers']

    def convert_data(self, base_dir):
        '''
        Enter PDF here
        PDFs normally have to be cleaned, but unfortunately
        due to the large variety in text, this process is not automated.
        Recommendation: Strip the title, index and references
        '''
        textbooks = [subj for subj in os.listdir(base_dir)
                     if os.path.isfile(
            os.path.join(base_dir, subj))]

        print("Converting all of these textbooks :", textbooks)

        converter = [PDFToTextConverter(remove_numeric_tables=False)
                     for i in textbooks]

        documents = [converter[i].convert(file_path=os.path.join(
            base_dir, textbooks[i])) for i in range(len(textbooks))]

        return documents

    def build_model(self):
        '''
        Automatic model builder, given the mode of operation
        Different retrievers, readers and pipelines can be supported as well
        '''
        # Experimental vs Actual
        if (self._exp):
            self._doc_store = InMemoryDocumentStore(similarity="dot_product")
        else:
            self._doc_store = FAISSDocumentStore(similarity="dot_product")

        self._processor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by='word',
            split_overlap=10,
            # Does not work while splitting by a passage
            split_respect_sentence_boundary=True
        )

        documents = self.convert_data(self._base_dir)

        doc_p = []
        for doc in documents:
            doc_p += self._processor.process(doc)

        print(type(doc_p[0]))

        self._doc_store.write_documents(doc_p)

        self._retriever = DensePassageRetriever(
            document_store=self._doc_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
        )

        self._doc_store.update_embeddings(self._retriever)

        # for poor guys like me who don't have a gpu
        gpu_yn = True if self._gpu == 1 else False

        if (self._speed):
            # prefer speed over acc, people are impatient
            self._reader = FARMReader(
                model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=gpu_yn)
        else:
            # prefer acc over speed, someone who likes to wait
            self._reader = FARMReader(
                model_name_or_path="deepset/minilm-uncased-squad2", use_gpu=gpu_yn)

    def add_pipeline(self, pipeline='default'):
        if (pipeline == 'default' or pipeline == 'read'):
            self._pipeline = ExtractiveQAPipeline(
                self._reader, self._retriever)

        elif (pipeline == 'retrive'):
            self._pipeline = DocumentSearchPipeline(
                self._retriever
            )
