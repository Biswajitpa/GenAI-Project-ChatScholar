import os
import re

from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter

# ==========================================
# OPENAI API KEY
# ==========================================

OPENAI_API_KEY = "
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

# ==========================================
# DATA FOLDER
# ==========================================

DATA_DIR = "__data__"
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        except Exception as e:
            print("PDF Error:", e)

    return text

# ==========================================
# TEXT CHUNKING
# ==========================================

def get_text_chunks(text):

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks

# ==========================================
# VECTOR STORE
# ==========================================

def get_vectorstore(chunks):

    if not chunks:
        raise ValueError("No text chunks found")

    try:

        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )

        return vectorstore

    except Exception as e:

        print("Embedding Error:", e)

        raise Exception(
            "OpenAI API Error. Check API key or billing."
        )

# ==========================================
# CONVERSATION CHAIN
# ==========================================

def get_conversation_chain(vectorstore):

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain

# ==========================================
# ESSAY GRADING
# ==========================================

def grade_essay(essay):

    global rubric_text

    try:

        messages = [
            {
                "role": "system",
                "content":
                "You are an AI essay grader. "
                "Grade according to rubric:\n"
                + rubric_text
            },
            {
                "role": "user",
                "content": essay
            }
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=1500
        )

        result = response.choices[0].message.content

        return re.sub(r"\n", "<br>", result)

    except Exception as e:

        print("Essay Error:", e)

        return "OpenAI API Error. Check billing or API key."

# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():

    return render_template('new_home.html')

# ==========================================
# PROCESS PDF
# ==========================================

@app.route('/process', methods=['POST'])
def process_documents():

    global vectorstore
    global conversation_chain

    try:

        pdf_docs = request.files.getlist('pdf_docs')

        if not pdf_docs or pdf_docs[0].filename == "":
            return "No PDF uploaded"

        raw_text = get_pdf_text(pdf_docs)

        if not raw_text.strip():
            return "No readable text found in PDF"

        text_chunks = get_text_chunks(raw_text)

        if len(text_chunks) == 0:
            return "No text chunks generated"

        vectorstore = get_vectorstore(text_chunks)

        conversation_chain = get_conversation_chain(vectorstore)

        return redirect('/chat')

    except Exception as e:

        print("Process Error:", e)

        return f"Error: {str(e)}"

# ==========================================
# CHAT PAGE
# ==========================================

@app.route('/chat', methods=['GET', 'POST'])
def chat():

    global chat_history
    global conversation_chain

    try:

        if request.method == 'POST':

            if conversation_chain is None:
                return "Please upload PDF first"

            user_question = request.form['user_question']

            response = conversation_chain.invoke({
                'question': user_question
            })

            chat_history = response['chat_history']

        return render_template(
            'new_chat.html',
            chat_history=chat_history
        )

    except Exception as e:

        print("Chat Error:", e)

        return f"Chat Error: {str(e)}"

# ==========================================
# PDF CHAT PAGE
# ==========================================

@app.route('/pdf_chat')
def pdf_chat():

    return render_template('new_pdf_chat.html')

# ==========================================
# ESSAY GRADING PAGE
# ==========================================

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():

    global rubric_text

    result = None
    text = ""

    try:

        if request.method == 'POST':

            # SAVE RUBRIC
            if request.form.get("essay_rubric"):

                rubric_text = request.form.get("essay_rubric")

                return render_template(
                    'new_essay_grading.html'
                )

            # PDF FILE
            if (
                'file' in request.files and
                request.files['file'].filename != ""
            ):

                pdf_file = request.files['file']

                text = extract_text_from_pdf(pdf_file)

            else:

                text = request.form.get("essay_text")

            if text:

                result = grade_essay(text)

            else:

                result = "No essay text found"

        return render_template(
            'new_essay_grading.html',
            result=result,
            input_text=text
        )

    except Exception as e:

        print("Essay Page Error:", e)

        return f"Essay Error: {str(e)}"

# ==========================================
# RUBRIC PAGE
# ==========================================

@app.route('/essay_rubric')
def essay_rubric():

    return render_template('new_essay_rubric.html')

# ==========================================
# PDF EXTRACT FUNCTION
# ==========================================

def extract_text_from_pdf(pdf_file):

    text = ""

    try:

        pdf_reader = PdfReader(pdf_file)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    except Exception as e:

        print("PDF Extraction Error:", e)

    return text

# ==========================================
# RUN APP
# ==========================================

if __name__ == '__main__':

    app.run(debug=True)
