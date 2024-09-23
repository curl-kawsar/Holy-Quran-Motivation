import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API key not found in environment variables. Please check 'key.env'.")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    নির্দেশাবলী:
    আপনি একটি AI সিস্টেম যা ব্যবহারকারীর অনুভূতি বুঝতে এবং পবিত্র কুরআন থেকে উদ্ধৃতি প্রদান করতে ডিজাইন করা হয়েছে। ব্যবহারকারী তাদের অনুভূতি বাংলায় ইনপুট করবে, যেমন বিষণ্ণতা বা উদ্বেগ। আপনার কাজ হল ব্যবহারকারীর মানসিক অবস্থা বোঝা এবং তাদের অনুপ্রাণিত ও সান্ত্বনা দেওয়ার জন্য পবিত্র কুরআনের প্রাসঙ্গিক আয়াত উল্লেখ করা।
    অনুগ্রহ করে:
    - বাংলায় ব্যবহারকারীর ইনপুটটি মনোযোগ সহকারে বিশ্লেষণ করুন।
    - ব্যবহারকারীর মানসিক অবস্থার সাথে সরাসরি সম্পর্কিত পবিত্র কুরআনের উদ্ধৃতি প্রদান করুন।
    - নিশ্চিত করুন যে আপনার কুরআনের উদ্ধৃতিগুলি সঠিক এবং প্রাসঙ্গিক।
    - যদি তথ্য অনুপস্থিত বা অস্পষ্ট হয়, তবে প্রসঙ্গের উপর ভিত্তি করে যৌক্তিক অনুমান করুন যাতে সর্বোত্তম অনুপ্রেরণা প্রদান করা যায়।
    - সংক্ষিপ্ত কিন্তু বিস্তারিত হন, প্রয়োজন হলে ব্যবহারকারীকে সান্ত্বনা দেওয়ার জন্য বিস্তারিত পদক্ষেপ প্রদান করুন।
    প্রসঙ্গ:\n{context}\n
    ব্যবহারকারীর অনুভূতি: \n{question}\n
    অনুপ্রেরণামূলক উদ্ধৃতি:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    with st.spinner('প্রসেসিং...'):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Get Motivation From the Holy AL-QURAN", page_icon="📚", layout="wide")
    st.header("📖 Get Motivation From the Holy AL-QURAN")


    Law = st.text_input("What Happened?")

    if Law:
        user_question = f"{Law}. And Please give me guideline, Motivation and instructions what will be best in this context from the Holy Quran, Give Precise instruction"
        user_input(user_question)

    with st.sidebar:
        st.title("Documents:")

        # Process the provided PDFs
        pdf_files = ["fab.pdf"]
        raw_text = get_pdf_text([open(pdf, "rb") for pdf in pdf_files])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

if __name__ == "__main__":
    main()
