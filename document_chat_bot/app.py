# import os
# import openai
# import json
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from langchain.document_loaders import TextLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from get_config import Config
# from docx import Document
# from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from langchain_community.document_loaders import TextLoader
# from flask import Flask, request, jsonify



import os
import openai
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from get_config import Config
from docx import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['GET'])
def greet():
    return "Hello"

@app.route('/api/v1/pre/processing', methods=['POST'])
def pre_process_file():
    req_body = request.json
    files = request.files.getlist('file')

    print("files:::::::::", files)

    return jsonify({
        "msg": "got the file"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port='5000')


Config = Config()

# with open('config.json', 'r') as file:
#     config_data = json.load(file)


os.environ["OPENAI_API_KEY"] = Config["open_ai"]["api_key"]


class Extractor:
    def __init__(self):
        self.openai_key = Config['open_ai']['api_key']
        self.openai_model = Config['open_ai']['embedding_model']
        self.openai_chat_model = Config['open_ai']['model']
        openai.api_key = self.openai_key   
        self.llm = ChatOpenAI(
            model = Config['open_ai']['model'],
            temperature = Config["open_ai"]["temperature"]
        )  


    def convert_docx_to_txt(self,docx_path, txt_path):
        """
        Convert a DOCX file to TXT.

        Parameters:
        - docx_path (str): Path to the input DOCX file.
        - txt_path (str): Path to save the output TXT file.
        """
        try:
            # Load the DOCX file
            doc = Document(docx_path)

            # Extract text from paragraphs
            text_content = []
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)

            # Save the text to a TXT file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write('\n'.join(text_content))
            
            print(f"Conversion successful. TXT file saved at: {txt_path}")

        except Exception as e:
            print(f"Error converting DOCX to TXT: {e}")
                 
        
    def convert_pdf_to_txt(self, pdf_path, txt_path):
        """
        Convert a PDF file to TXT.

        Parameters:
        - pdf_path (str): Path to the input PDF file.
        - txt_path (str): Path to save the output TXT file.
        """
        try:
            loader = PyPDFLoader(pdf_path)
            document = loader.load()

            # Save the text to a TXT file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(document)

            print(f"Conversion successful. TXT file saved at: {txt_path}")

        except Exception as e:
            print(f"Error converting PDF to TXT: {e}")

    
    def plan_embedding_with_open_ai(self, plan_file):
        try:
            if plan_file.lower().endswith('.docx'):
                loader = Docx2txtLoader(plan_file)
            elif plan_file.lower().endswith('.pdf'):
                loader = PDFLoader(plan_file)
            else:
                return "Unsupported file format. Only DOCX and PDF files are supported."

            documents = loader.load()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=500
            )
            document_chunks = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key, model=self.openai_model)

            vector_db = FAISS.from_documents(document_chunks, embeddings)

            return vector_db
        except Exception as e:
            print("Error in plan_embedding function", e)

    def prompt_executor_with_open_ai(self, query, vector_db):

        try:
            context = vector_db.similarity_search(query=prompt, search_type="similarity")
            user_prompt_template = """

                Answer the question based only on the query:
                    context : {context}
                    query : {query}

                    Note:
                    1. You MUST extract the value and return only the value.
                    2. You MUST not give descriptive answers
                    3. You MUST only return the value or cost
                    4. If the user input asks for descriptive answer ignore note number 1,2 and 3
                    5. Make sure not to add any quotation in response returned
                    3. Simply return the output as string do not add any leading and trailing texts
            """

            prompt_template = PromptTemplate(
                template=user_prompt_template, input_variables=["context", "query"]
            )


            chain = LLMChain(llm = self.llm, prompt=prompt_template)

            input_variables = {
                "context" : context,
                "query" : query
            }

            llmRes = chain(input_variables)

            return llmRes['text']

        except Exception as e:
            print("Error in prompt_executor_with_open_ai", e)



ex = Extractor()
file_path = "./Vishal_resume.pdf"
# txt_path = "./Sample_Payer_Contract.txt"
# ex.convert_docx_to_txt(file_path, txt_path)

vector_db = ex.plan_embedding_with_open_ai(file_path)


prompt = "what are the skills of Vishal Kumar Yadav"
res = ex.prompt_executor_with_open_ai(prompt, vector_db)

print("res:::::::::::::",res)


# with open('prompts.json', 'r') as file:
#     # Load the JSON data
#     data = json.load(file)

# for key, value in data.items():
#     print(f"{key}:::::::::::::::::::::::::::::::::::::::{value}\n\n")
