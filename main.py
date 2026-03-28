from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use ONLY the provided context. If answer not found, say you don't know."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)

print("✅ RAG system created")
print("👉 Press 0 to exit\n")

while True:
    query=input("You : ")

    if query == "0":
        print("EXIT....")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        # LLM response
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")

    if not docs:
        print("\nAI: No relevant documents found.\n")
        continue
