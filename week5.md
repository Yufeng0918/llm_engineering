## Week 5 Learning Notes: Building an Expert Knowledge Worker with RAG

This week's focus is on constructing an "Expert Knowledge Worker"—a question-answering agent specifically designed for Insurellm, an Insurance Tech company. The core principles guiding this development are accuracy and cost-efficiency, primarily achieved through the implementation of Retrieval Augmented Generation (RAG).

### Foundational RAG Implementation

**Concept: Retrieval Augmented Generation (RAG)**
RAG is a technique that enhances the capabilities of Large Language Models (LLMs) by allowing them to access and leverage external, up-to-date, and domain-specific information during text generation. Instead of relying solely on the knowledge embedded in their training data, RAG models first retrieve relevant documents or data snippets from a knowledge base and then use this retrieved information to inform their responses. This approach significantly improves accuracy, reduces hallucinations, and enables LLMs to answer questions about proprietary or rapidly changing information.

**Initial Brute-Force RAG Approach**
The initial implementation focuses on a straightforward RAG mechanism.

*   **Model Selection**: The `gpt-4o-mini` model is chosen due to its balance of performance and low cost, aligning with the company's requirements.

    ```python
    MODEL = "gpt-4o-mini"
    ```

*   **Data Ingestion**: Markdown files containing information about employees and products are loaded from the `knowledge-base` directory. This content is then stored in a Python dictionary, where keys represent entities (e.g., employee names, product names) and values hold the corresponding document text.

    ```python
    import os
    import glob
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
    openai = OpenAI()

    context = {}

    employees = glob.glob("knowledge-base/employees/*")
    for employee in employees:
        name = employee.split(os.sep)[-1][:-3] # Handles both Windows and Unix paths
        doc = ""
        with open(employee, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name]=doc

    products = glob.glob("knowledge-base/products/*")
    for product in products:
        name = product.split(os.sep)[-1][:-3] # Handles both Windows and Unix paths
        doc = ""
        with open(product, "r", encoding="utf-8") as f:
            doc = f.read()
        context[name]=doc
    ```

*   **Agent Persona and System Message**: A `system_message` defines the agent's role: an expert in Insurellm, providing concise and accurate answers, and explicitly stating when information is not available.

    ```python
    system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."
    ```

*   **Context Retrieval and Augmentation**:
    *   `get_relevant_context`: This function performs a simple keyword-based search. If a `context_title` (e.g., "Lancaster") is found within the user's message, its associated document content is retrieved.
    *   `add_context`: This function appends the retrieved relevant context to the user's original message before it's sent to the LLM. This allows the LLM to use the specific, relevant information to formulate its answer.

    ```python
    def get_relevant_context(message):
        relevant_context = []
        for context_title, context_details in context.items():
            if context_title.lower() in message.lower():
                relevant_context.append(context_details)
        return relevant_context

    def add_context(message):
        relevant_context = get_relevant_context(message)
        if relevant_context:
            message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
            for relevant in relevant_context:
                message += relevant + "\n\n"
        return message
    ```

*   **Chat Interface**: Gradio's `ChatInterface` is employed for rapid prototyping of the conversational agent.

    ```python
    import gradio as gr

    def chat(message, history):
        messages = [{"role": "system", "content": system_message}] + history
        message = add_context(message)
        messages.append({"role": "user", "content": message})

        stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            yield response

    view = gr.ChatInterface(chat, type="messages").launch()
    ```

### Enhancing Document Processing with LangChain

**Concept: LangChain**
LangChain is a framework designed to simplify the development of applications powered by language models. It provides abstractions for common components (like LLMs, prompt templates, chains, and agents) and allows developers to chain these components together to create more complex applications, such as RAG systems, chatbots, and agents that can interact with external tools.

**Document Loading and Splitting**
To manage larger knowledge bases and prepare data for more sophisticated RAG, LangChain's document handling capabilities are introduced.

*   **Document Loading**: `DirectoryLoader` is used to efficiently load all markdown files (`.md`) recursively from the `knowledge-base` directory. `TextLoader` is specified as the loader class, and `encoding='utf-8'` is used to ensure proper text interpretation. Metadata, specifically `doc_type` (e.g., 'employees', 'products'), is added to each document based on its source folder, which is crucial for later filtering and analysis.

    ```python
    from langchain.document_loaders import DirectoryLoader, TextLoader
    import os
    import glob

    folders = glob.glob("knowledge-base/*")
    text_loader_kwargs = {'encoding': 'utf-8'}

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    ```

*   **Text Splitting**: `CharacterTextSplitter` is employed to break down large documents into smaller, more manageable `chunks`. This is vital because LLMs often have token limits, and retrieving smaller, highly relevant chunks is more efficient than retrieving entire large documents. `chunk_size` and `chunk_overlap` parameters control the size of these chunks and the overlap between them, which helps maintain context across splits.

    ```python
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    ```
    *Note: For systems with limited memory, adjusting `chunk_size` (e.g., to 2000) and `chunk_overlap` (e.g., to 400) might be necessary to prevent crashes.*

### Vector Stores, Embeddings, and Visualization

**Concept: Embeddings**
Embeddings are numerical representations (vectors) of text, images, or other data that capture their semantic meaning. In the context of text, words or phrases with similar meanings are mapped to points that are close to each other in a high-dimensional vector space. These numerical representations allow algorithms to understand and process natural language, enabling tasks like semantic search, clustering, and recommendation systems.

**Concept: Vector Store (Vector Database)**
A vector store (or vector database) is a specialized database designed to efficiently store, manage, and query vector embeddings. Unlike traditional databases that store structured data, vector stores are optimized for similarity searches, allowing quick retrieval of vectors that are "closest" (most similar) to a given query vector. This is fundamental for RAG, as it enables the system to rapidly find the most semantically relevant document chunks for a given user query.

**Implementation with OpenAI Embeddings and ChromaDB**

*   **Embedding Generation**: `OpenAIEmbeddings` is used to convert the text `chunks` into high-dimensional vector embeddings. This model is an example of an "Auto-Encoding LLM," which generates an output (the embedding) given a complete input. This contrasts with "Auto-Regressive LLMs" (like GPT models) that generate future tokens based on past context.

    ```python
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    # Optional: Use open-source embeddings (e.g., for data privacy)
    # from langchain.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ```

*   **Chroma Vector Store**: `Chroma`, a popular open-source vector database, is used to store these embeddings. The `persist_directory` parameter ensures that the vector store data is saved to disk, allowing it to be reused across sessions.

    ```python
    from langchain_chroma import Chroma
    import os

    db_name = "vector_db"

    # Delete existing collection to start fresh
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    ```

*   **Vector Store Inspection**: The number of vectors and their dimensionality can be inspected. OpenAI embeddings typically have 1536 dimensions.

    ```python
    collection = vectorstore._collection
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"The vectors have {dimensions:,} dimensions")
    ```

**Visualizing the Vector Space**
To better understand how documents are semantically related and clustered in the high-dimensional vector space, dimensionality reduction and visualization techniques are applied.

*   **t-SNE (t-distributed stochastic neighbor embedding)**: This algorithm is used to reduce the high-dimensional embeddings to 2D and 3D, making them visualizable. t-SNE is particularly good at preserving local relationships, meaning points that are close in the high-dimensional space will tend to remain close in the reduced-dimensional space.
*   **Plotly Visualization**: `plotly.graph_objects` creates interactive scatter plots. Points are colored based on their `doc_type` (e.g., employees, products), allowing visual identification of clusters of similar document types. Hover information displays the original text and document type for context.

    ```python
    import numpy as np
    from sklearn.manifold import TSNE
    import plotly.graph_objects as go

    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
    color_map = {'products':'blue', 'employees':'green', 'contracts':'red', 'company':'orange'}
    colors = [color_map[t] for t in doc_types]

    # 2D Visualization
    tsne_2d = TSNE(n_components=2, random_state=42)
    reduced_vectors_2d = tsne_2d.fit_transform(vectors)

    fig_2d = go.Figure(data=[go.Scatter(
        x=reduced_vectors_2d[:, 0],
        y=reduced_vectors_2d[:, 1],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])
    fig_2d.update_layout(title='2D Chroma Vector Store Visualization', width=800, height=600)
    fig_2d.show()

    # 3D Visualization
    tsne_3d = TSNE(n_components=3, random_state=42)
    reduced_vectors_3d = tsne_3d.fit_transform(vectors)

    fig_3d = go.Figure(data=[go.Scatter3d(
        x=reduced_vectors_3d[:, 0],
        y=reduced_vectors_3d[:, 1],
        z=reduced_vectors_3d[:, 2],
        mode='markers',
        marker=dict(size=5, color=colors, opacity=0.8),
        text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
        hoverinfo='text'
    )])
    fig_3d.update_layout(title='3D Chroma Vector Store Visualization', width=900, height=700)
    fig_3d.show()
    ```

### Building Conversational RAG with LangChain

**Concept: ConversationalRetrievalChain**
In LangChain, the `ConversationalRetrievalChain` is a powerful tool for building chatbots that can answer questions based on a knowledge base while also maintaining a conversational history. It combines an LLM, a retriever (which fetches relevant documents from a vector store), and a memory component to handle multi-turn conversations. When a new question is asked, the chain considers both the current question and the chat history to retrieve the most relevant context before generating a response.

**Integrating Components for a Chatbot**

*   **LLM Initialization**: A `ChatOpenAI` instance is created, specifying the LLM model (`gpt-4o-mini`) and temperature (creativity).

    ```python
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    # Optional: Use Ollama for local LLM inference
    # llm = ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')
    ```

*   **Conversation Memory**: `ConversationBufferMemory` is set up to store the chat history. `memory_key='chat_history'` and `return_messages=True` are crucial for the chain to access past turns in the conversation.

    ```python
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Note: A LangChainDeprecationWarning might appear regarding this memory usage.
    # For simplicity, it is currently ignored as migrating to LangGraph is more complex.
    ```

*   **Retriever Setup**: The `vectorstore` (Chroma or FAISS) is converted into a `retriever` instance. This abstraction allows the `ConversationalRetrievalChain` to query the vector store for relevant document chunks. The `search_kwargs={"k": 25}` parameter in `as_retriever()` specifies that the retriever should fetch the top 25 most relevant document chunks.

    ```python
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    ```

*   **Chain Construction**: The `ConversationalRetrievalChain.from_llm` method brings all these components together: the LLM, the retriever, and the memory. This chain handles the end-to-end process of receiving a query, retrieving context, incorporating chat history, and generating a coherent response.

    ```python
    from langchain.chains import ConversationalRetrievalChain

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    ```

*   **Gradio Chat Interface**: The `conversation_chain` is wrapped in a `chat` function and exposed via Gradio for interactive use.

    ```python
    import gradio as gr

    def chat(question, history): # history parameter is present for Gradio, but memory is handled by conversation_chain
        result = conversation_chain.invoke({"question": question})
        return result["answer"]

    view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
    ```

*   **Debugging with Callbacks**: LangChain's `StdOutCallbackHandler` can be used to observe the internal workings of the `conversation_chain`, printing out the steps and intermediate results, which is invaluable for debugging and understanding the RAG process.

    ```python
    from langchain_core.callbacks import StdOutCallbackHandler

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])
    ```

### Alternative Vector Store: FAISS

**Concept: FAISS (Facebook AI Similarity Search)**
FAISS is a library for efficient similarity search and clustering of dense vectors. It provides algorithms that search for points in a vector space that are closest to a given query point. FAISS is known for its high performance and scalability, making it a popular choice for large-scale similarity search tasks, including those in RAG systems.

*   **Switching from Chroma to FAISS**: The `vectorstore` creation can be easily swapped from ChromaDB to FAISS, demonstrating the modularity of LangChain. The rest of the RAG pipeline (LLM, memory, `ConversationalRetrievalChain`) remains unchanged, highlighting the interchangeability of vector store implementations.

    ```python
    # BEFORE (Chroma)
    # from langchain_chroma import Chroma
    # vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

    # AFTER (FAISS)
    from langchain.vectorstores import FAISS
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    total_vectors = vectorstore.index.ntotal
    dimensions = vectorstore.index.d
    print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")
    ```

The visualization logic remains largely the same, confirming that the plots represent the embeddings themselves, rather than being specific to the chosen vector store.

---

This comprehensive overview of Week 5's content provides a step-by-step guide to building an expert knowledge worker using RAG, covering fundamental concepts, practical implementations with LangChain, and considerations for scalability and deployment.
   