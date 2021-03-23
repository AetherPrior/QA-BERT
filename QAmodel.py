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

# Different pipelines for each model
from haystack.pipeline import ExtractiveQAPipeline, DocumentSearchPipeline, GenerativeQAPipeline


class QAmodel:
    def __init__(self, base_dir,
                 exp=False,
                 speed=True,
                 gpu=False,
                 reader_dir=None,
                 retriever_dir=None,
                 doc_file=None,
                 save=True):

        # data for answering query
        self._base_dir = base_dir

        # model (built?)
        self._model = False

        # pipeline
        self._pipeline = None

        # components of the pipeline
        self.textbooks = None

        self._converter = None
        self._doc_store = None
        self._processor = None
        self._retriever = None
        self._reader = None

        # modes
        self._exp, self._speed, self._gpu, self._save = exp, speed, gpu, save

        # directories
        self.reader_dir = reader_dir
        self.retriever_dir = retriever_dir

        '''
        An open github issue ( https://github.com/deepset-ai/haystack/issues/135 )
        is causing some unexpected results
        This feature is blocked for now
        self.doc_file = doc_dir
        '''

    def get_tb_string(self):
        '''
        Helper method to deal with document saving
        '''

        return "".join(i[:-4] for i in self._textbooks)+".pt"

    def execute_query(self, query):
        '''
        We attempt to find the answer to a query from the retriever based the model
        prerequisite: The model is built
        postcondition: The answer(s) is(are) generated and returned
        '''

        if (not self._model):
            print(
                f"{bcolors.WARNING} A model has not been built! We can build it if you wish: [y/N] {bcolors.ENDC}")
            opt = input()
            if (opt.lower() == 'y'):
                self.build_model()
                self.add_pipeline()
            else:
                return

        elif(self._pipeline is None):
            print(
                f"{bcolors.WARNING}A pipeline has not been set! We can make it if you wish: [y/N]{bcolors.ENDC}")
            opt = input()
            if (opt.lower() == 'y'):
                self.build_model()
                self.add_pipeline()
            else:
                return

        prediction = self._pipeline.run(query=query,
                                        top_k_retriever=5, top_k_reader=5)

        return prediction

    def convert_data(self, base_dir):
        '''
        Enter PDF here
        PDFs normally have to be cleaned, but unfortunately
        due to the large variety in text, this process is not automated.
        Recommendation: Strip the title, index and references
        TBD, add support for docs
        '''

        self._textbooks = [subj for subj in os.listdir(base_dir)
                           if os.path.isfile(
            os.path.join(base_dir, subj))]

        print(
            f"{bcolors.OKGREEN}Converting all of these textbooks :{self._textbooks}{bcolors.ENDC}")

        converter = [PDFToTextConverter(remove_numeric_tables=False)
                     for i in self._textbooks]

        documents = [converter[i].convert(file_path=os.path.join(
            base_dir, self._textbooks[i])) for i in range(len(self._textbooks))]

        return documents

    def build_model(self):
        '''
        Automatic model builder, given the mode of operation
        Different readers and pipelines (tbd) can be supported as well
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

        self._doc_store.write_documents(doc_p)

        if (self.retriever_dir is not None and os.path.isdir("./"+self.retriever_dir)):

            self._retriever = DensePassageRetriever(
                document_store=self._doc_store,
                query_embedding_model=self.retriever_dir+"/query_encoder",
                passage_embedding_model=self.retriever_dir+"/passage_encoder"
            )

            self._retriever.load(load_dir=self.retriever_dir,
                                 document_store=self._doc_store)

        else:

            if(self.retriever_dir is None):
                print(
                    f"{bcolors.OKCYAN}No retriever directory specified, (re)starting model{bcolors.ENDC}")
            else:
                print(
                    f"{bcolors.WARNING}No such retriever directory: {self.retriever_dir}{bcolors.ENDC}"
                )
            self._retriever = DensePassageRetriever(
                document_store=self._doc_store,
                query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
            )

        self._doc_store.update_embeddings(self._retriever)

        # for poor guys like me who don't have a gpu
        gpu_yn = self._gpu

        # Currently defined 2 readers
        speed_reader = "deepset/minilm-uncased-squad2"
        non_fast = "deepset/roberta-base-squad2"

        if (self.reader_dir is not None and os.path.isdir("./"+self.reader_dir)):
            self._reader = FARMReader(model_name_or_path=self.reader_dir)

        else:
            if(self.reader_dir is None):
                print(
                    f"{bcolors.OKCYAN}No reader directory specified, (re)starting model{bcolors.ENDC}")
            else:
                print(
                    f"{bcolors.WARNING}No such reader directory:{bcolors.ENDC}"
                )
            if (self._speed):
                # prefer speed over acc, people are impatient
                self._reader = FARMReader(
                    model_name_or_path=speed_reader, use_gpu=gpu_yn)
            else:
                # prefer acc over speed, someone who likes to wait
                self._reader = FARMReader(
                    model_name_or_path=non_fast, use_gpu=gpu_yn)

        if(not self._exp and self._save):

            self._retriever.save('models/retrievers/dpr/',
                                 query_encoder_dir='query_encoder/',
                                 passage_encoder_dir='passage_encoder/'
                                 )

            tbstring = "models/documents/"+self.get_tb_string()

            # self._doc_store.save(tbstring) # Feature blocked till github issue is resolved for haystack
            reader = speed_reader if self._speed == True else non_fast
            self._reader.save('models/readers/FARMreader/'+speed_reader)

        self._model = True

    def add_pipeline(self, pipeline='default'):
        '''
        ....makes a pipeline from the haystack defaults
        TBD: Support generative pipelines and custom ones. 
        '''

        if (pipeline == 'default' or pipeline == 'read'):
            self._pipeline = ExtractiveQAPipeline(
                self._reader, self._retriever)

        elif (pipeline == 'retrive'):
            self._pipeline = DocumentSearchPipeline(
                self._retriever
            )
