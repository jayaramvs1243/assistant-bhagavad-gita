import logging
import os
import sys
from time import sleep

import streamlit as st
from llama_index.core import SimpleDirectoryReader, ChatPromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import models, QdrantClient
from llama_index.core.llms import ChatMessage, MessageRole

import app.utilities.message_template


message_template_2 = [
    ChatMessage(
        content="""
        You are an expert ancient assistant who is well versed in Bhagavad-gita.
        You are Multilingual, you understand English, Hindi and Sanskrit.

        Always structure your response in this format:
        <think>
        [Your step-by-step thinking process here]
        </think>

        [Your final answer here]
        
        We have provided context information below.
        {context_str}
        ---------------------
        Given this information, please answer the question: {query}
        ---------------------
        If the question is not from the provided context, say `I don't know. Not enough information received.`
        """,
        role=MessageRole.USER,
    ),
]


@st.cache_resource
def initialize_embedding_model() -> BaseEmbedding:
    embedding_model_name = "thenlper/gte-large"

    embedding_model = FastEmbedEmbedding(model_name=embedding_model_name)
    return embedding_model


@st.cache_resource
def initialize_vector_client() -> QdrantClient:
    qdrant_url = "https://98a27e1e-d49f-4ae4-bbe8-88494b4628af.eu-west-1-0.aws.cloud.qdrant.io"
    qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MDUyMDQyfQ._-VLuwhLETrQIywI9nmd9xWmaf5HRcHdHiyIYvFghxE"

    vector_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )
    return vector_client


@st.cache_resource
def initialize_ll_model() -> LLM:
    groq_api_key = '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'
    os.environ['GROQ_API_KEY'] = groq_api_key
    
    ll_model_name = "deepseek-r1-distill-llama-70b"

    ll_model = Groq(model=ll_model_name)
    return ll_model


def vectorize(embedding_model: BaseEmbedding, vector_client: QdrantClient, collection_name: str):
    dataset_dir = "../../dataset"

    reader = SimpleDirectoryReader(
        input_dir=dataset_dir,
        required_exts=[".pdf"],
        recursive=True
    )
    print(f"Number of files found in the directory {dataset_dir}: {len(reader.input_files)}")

    documents = reader.load_data()
    print(f"Number of documents loaded from the directory {dataset_dir}: {len(documents)}")

    print("Extracting the text content from the list of documents.")
    document_contents = [document.text for document in documents]

    if not vector_client.collection_exists(collection_name=collection_name):
        vector_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE, on_disk=True),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True),
            )
        )
    else:
        raise Exception("Collection already exists in vector database.")

    batch_size = 50
    print(
        f"Generating embeddings of the text in chunks of size {batch_size} using the embedding model {embedding_model.model_name}")
    print(f"Uploading the embeddings to the vector database collection '{collection_name}'")
    for page in range(0, len(document_contents), batch_size):
        page_content = document_contents[page:page + batch_size]
        embeds = embedding_model.get_text_embedding_batch(page_content)

        vector_client.upload_collection(
            collection_name=collection_name,
            vectors=embeds,
            payload=[{"context": content} for content in page_content]
        )

    vector_client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )


def search(embedding_model: BaseEmbedding, vector_client: QdrantClient, collection_name: str, user_query: str, k=5):
    user_query_embedding = embedding_model.get_query_embedding(user_query)
    result = vector_client.query_points(
        collection_name=collection_name,
        query=user_query_embedding,
        limit=k
    )
    return result


def pipeline(embedding_model: BaseEmbedding, vector_client: QdrantClient, collection_name: str, conversation_history: str, user_query: str,
             ll_model: LLM):
    # R - Retriever
    relevant_documents = search(embedding_model, vector_client, collection_name, user_query)
    query_context = [doc.payload["context"] for doc in relevant_documents.points]
    query_context = "\n".join(query_context)

    # Combine context from retrieved documents and conversation history
    full_conversation_context = f"{conversation_history}\n\nContext from documents:\n{query_context}"

    # A - Augment
    chat_template = ChatPromptTemplate(message_templates=app.utilities.message_template.message_template_2)

    # G - Generate
    response = ll_model.complete(
        chat_template.format(
            context_str=full_conversation_context,
            query=user_query)
    )
    return response


def extract_thinking_and_answer(response_text):
    """
    Extract thinking process and final answer from response
    """
    try:
        thinking = response_text[response_text.find("<think>") + 7:response_text.find("</think>")].strip()
        answer = response_text[response_text.find("</think>") + 8:].strip()
        return thinking, answer
    except:
        return "", response_text


def main():
    st.title("🕉️ Bhagavad Gita Assistant")

    collection_name = "bhagavad-gita"

    # This will run only once, and be saved inside the cache
    embedding_model = initialize_embedding_model()
    vector_client = initialize_vector_client()
    ll_model = initialize_ll_model()

    if not vector_client.collection_exists(collection_name):
        # To read the documents from the source directory,
        # create vector representation (embeddings) and store to the vector database
        vectorize(embedding_model, vector_client, collection_name)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                thinking, answer = extract_thinking_and_answer(message["content"])
                with st.expander("Show thinking process"):
                    st.markdown(thinking)
                st.markdown(answer)
            else:
                st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("Ask your question about the Bhagavad Gita..."):
        # Display user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                # Build conversation history from past messages for full context
                conversation_history = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
                )

                full_response = pipeline(embedding_model, vector_client, collection_name, conversation_history, user_query, ll_model)
                thinking, answer = extract_thinking_and_answer(full_response.text)

                with st.expander("Show thinking process"):
                    st.markdown(thinking)

                response = ""
                for chunk in answer.split():
                    response += chunk + " "
                    message_placeholder.markdown(response + "▌")
                    sleep(0.05)

                message_placeholder.markdown(answer)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response.text})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    main()
