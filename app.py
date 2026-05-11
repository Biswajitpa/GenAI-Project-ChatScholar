import os
import re
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

# =========================
# LANGCHAIN (FIXED IMPORTS)
# =========================
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter

# =========================
# OPENAI SDK
# =========================
from openai import OpenAI

# =========================
# API KEY
# =========================
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []
rubric_text = ""

DATA_DIR = "__data__"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# PDF TEXT EXTRACTION
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# =========================
# TEXT CHUNKING
# =========================
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# =========================
# VECTOR STORE (FAISS)
# =========================
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# =========================
# CONVERSATION CHAIN
# =========================
def get_conversation_chain(vs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(),
        memory=memory
    )

# =========================
# ESSAY GRADING FUNCTION
# =========================
def _grade_essay(essay):
    messages = [
        {
            "role": "system",
            "content": "You are an AI grader. Grade strictly based on the rubric:\n" + rubric_text
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
    return re.sub(r'\n', '<br>', result)

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return render_template('new_home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain

    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)

    chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    return redirect('/chat')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chat_history, conversation_chain

    if request.method == 'POST':
        user_question = request.form['user_question']
        response = conversation_chain({"question": user_question})
        chat_history = response["chat_history"]

    return render_template('new_chat.html', chat_history=chat_history)


@app.route('/pdf_chat')
def pdf_chat():
    return render_template('new_pdf_chat.html')


@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    global rubric_text

    result = None
    text = ""

    if request.method == 'POST':

        # Save rubric
        if request.form.get("essay_rubric"):
            rubric_text = request.form.get("essay_rubric")
            return render_template('new_essay_grading.html')

        # PDF input
        if 'file' in request.files and request.files['file'].filename != "":
            text = extract_text_from_pdf(request.files['file'])
        else:
            text = request.form.get("essay_text")

        result = _grade_essay(text)

    return render_template(
        'new_essay_grading.html',
        result=result,
        input_text=text
    )


@app.route('/essay_rubric')
def essay_rubric():
    return render_template('new_essay_rubric.html')

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)