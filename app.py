import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads and caches the FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS vector store from {DB_FAISS_PATH}: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Creates a PromptTemplate object."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Loads the HuggingFace language model."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512 
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.stop() 

    # Define custom prompt template
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer.
    Dont provide anything out of the given context

    Context: {context}
    Question: {input}

    Start the answer directly. No small talk please.
    """

    # HuggingFace model configuration
    # from dotenv import load_dotenv
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN") # Ensure HF_TOKEN is set as an environment variable

    # Get user query
    prompt_input = st.chat_input("Pass your prompt here")

    if prompt_input:
        # Display user message
        st.chat_message('user').markdown(prompt_input)
        st.session_state.messages.append({'role':'user', 'content': prompt_input})

        try:
            # 1. Load LLM and set up custom prompt
            llm_instance = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
            prompt_instance = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            # 2. Create the chain that combines documents (using your LLM and custom prompt)
            combine_docs_chain = create_stuff_documents_chain(
                llm_instance,
                prompt_instance
            )

            # 3. Create the retrieval chain
            # It combines the retriever (from vectorstore) with the document combining chain
            qa_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={'k':3}), # Retrieve top 3 documents
                combine_docs_chain
            )

            # Invoke the RAG chain with the user's query
            response = qa_chain.invoke({'input': prompt_input})

            # Extract result and source documents
            result = response["answer"] 
            source_documents = response["context"] 

            result_to_show = f"{result}\n\n**Source Documents:**\n"
            if source_documents:
                for doc in source_documents:
                    result_to_show += f"- {doc.metadata.get('source', 'Unknown Source')}: {doc.page_content[:150]}...\n" # Show snippet
            else:
                result_to_show += "No relevant source documents found."

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error during RAG chain invocation: {e}")
            st.write("Please ensure your `HF_TOKEN` environment variable is set and correct.")

if __name__ == "__main__":
    main()