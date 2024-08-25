
import PyPDF2
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
embeddingsModel = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

 
def textFromPDF(file) -> list:
    # Open a PDF file
    # with open('example2.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)

    # Get the number of pages in the PDF
    num_pages = len(reader.pages)
    print(f'Total pages: {num_pages}')

    # Extract text from the all page
    text = ''
    for i in range(num_pages):
        text += reader.pages[i].extract_text()
    return [text]
    
def explitTextIntoChunks(text: list):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    is_separator_regex=False,
    )
    texts = text_splitter.create_documents(text)
    return texts

def createVectorOfText(text) -> None:
    # Convert the text to LangChain's `Document` format
    # docs =  [Document(page_content=text, metadata={"source": "local"})]
    docs = text
    db = FAISS.from_documents(docs, embeddingsModel)
    db.save_local("vectorIndex")
    return True

def searchYourQuery(query):
    db = FAISS.load_local(folder_path="vectorIndex",embeddings=embeddingsModel,allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)
    print("1st search")
    print(docs[0].page_content)
    return docs
    # print("2nd search")
    # embedding_vector = embeddingsModel.embed_query(query)
    # print(len(embedding_vector))
    # docs = db.similarity_search_by_vector(embedding_vector)
    # print(docs[0].page_content)



import streamlit as st

# Create containers for left and right sections
left_col = st.container()
right_col = st.container()

# Adjust the layout with fixed widths for left and right columns
with left_col:
    st.markdown("""
    <style>
    .left-col {
        float: left;
        width: 30%;
        padding: 10px;
        box-sizing: border-box;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="left-col">', unsafe_allow_html=True)
    st.markdown("## PDF Upload & Index Creation")
    st.markdown("---")  # Adds a horizontal line for separation

    if "index_created" not in st.session_state:
        st.session_state.index_created = False

    # File uploader allowing multiple file selections
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Create Index"):
            with st.spinner('Creating index from the uploaded PDFs...'):
                # Process each PDF file and create an index
                for uploaded_file in uploaded_files:
                    text = textFromPDF(uploaded_file)
                    chunk = explitTextIntoChunks(text)
                    createVectorOfText(chunk)  # Assume this function handles multiple PDFs

            st.session_state.index_created = True
            st.success('Index created successfully!')

    if st.session_state.index_created:
        st.success('Index already created! You can create more indexes anytime.')
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <style>
    .right-col {
        float: right;
        width: 50%;
        padding: 10px;
        box-sizing: border-box;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="right-col">', unsafe_allow_html=True)
    st.markdown("## Query the Index")
    st.markdown("---")  # Adds a horizontal line for separation
    
    if st.session_state.index_created:
        query = st.text_input("Enter your query:")
        
        if query:
            # Perform the query on the created index
            result = searchYourQuery(query)
            st.write(f"**Most relevant result:** {result}")
    else:
        st.warning("Please create an index from the PDFs before querying.")
    st.markdown('</div>', unsafe_allow_html=True)