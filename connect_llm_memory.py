import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain # NEW IMPORT
# from dotenv import load_dotenv # NEW IMPORT

# --- Configuration ---
# Ensure your HuggingFace API token is set as an environment variable (e.g., HF_TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Utility Functions ---
def load_llm(huggingface_repo_id, hf_token): # Added hf_token parameter
    """Loads the HuggingFace language model."""
    if not hf_token:
        raise ValueError("HuggingFace API token (HF_TOKEN) not found in environment variables.")
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=hf_token, # Use the passed hf_token
        max_new_tokens=512
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Creates a PromptTemplate object."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# --- Main Script Execution ---
if __name__ == "__main__":
    # Load LLM
    try:
        llm_instance = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your HuggingFace API token as an environment variable (HF_TOKEN).")
        exit() # Exit if token is not found

    # Set custom prompt
    prompt_instance = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    # Load Embedding Model and FAISS Database (CPU-only)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS vector store from {DB_FAISS_PATH}: {e}")
        print("Please ensure the 'vectorstore/db_faiss' directory exists and contains valid FAISS index files.")
        exit()

    # --- QA Chain Creation (Updated for LangChain 0.2.x API) ---
    # 1. Create the chain that combines documents (using your LLM and custom prompt)
    combine_docs_chain = create_stuff_documents_chain(
        llm_instance,
        prompt_instance
    )

    # 2. Create the retrieval chain (combines the retriever with the document combining chain)
    qa_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={'k':3}), # Retriever to get top 3 documents
        combine_docs_chain                       # Chain to combine those documents with the prompt
    )

    # --- Invoke with a query ---
    user_query = input("Write Query Here: ")

    try:
        response = qa_chain.invoke({'input': user_query})

        # --- Accessing results (Updated for LangChain 0.2.x response keys) ---
        print("\n--- RESULT ---")
        print(response.get("answer", "No answer found.")) # LangChain 0.2.x uses "answer"

        print("\n--- SOURCE DOCUMENTS ---")
        source_documents = response.get("context", []) # LangChain 0.2.x uses "context"
        if source_documents:
            for i, doc in enumerate(source_documents):
                print(f"Document {i+1}:")
                print(f"  Source: {doc.metadata.get('source', 'N/A')}")
                print(f"  Page Content (Snippet): {doc.page_content[:200]}...") # Print a snippet
                print("-" * 20)
        else:
            print("No relevant source documents found.")

    except Exception as e:
        print(f"An error occurred during query invocation: {e}")