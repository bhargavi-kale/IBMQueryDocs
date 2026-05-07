from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def run_rag_pipeline(question, retriever, rerank_model, client):
    # --- Step A: Retriever logic with Reranking ---
    def get_reranked_docs(query):
        initial_docs = retriever.invoke(query)
        if not initial_docs: return []
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = rerank_model.predict(pairs)
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:3]]

    # --- Step B: Format function from your image ---
    def format_docs(docs):
        """Standardizes document formatting with Source IDs."""
        return "\n\n".join(
            f"DOCUMENT ID: {d.metadata.get('source')}\nCONTENT: {d.page_content}" 
            for d in docs
        )

    # --- Step C: Chat Prompt Template ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "COMMAND: Answer ONLY using provided DOCUMENTS. Append [Source: ID] after every sentence. If missing, say 'The provided documentation does not contain information about this.'"),
        ("user", "DOCUMENTS:\n{context}\n\nQUESTION: {question}")
    ])

    # --- Step D: The LLM 'Runnable' wrapper ---
    def llm_chain_call(prompt_value):
        """Bypasses 'text-generation' errors by using chat_completion."""
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt_value.to_string()}],
            max_tokens=500,
            temperature=0 # Strict factual extraction
        )
        return response.choices[0].message.content

    # --- Step E: The LCEL Chain ---
    # This exactly matches the flow in your image
    rag_chain = (
        {
            "context": RunnableLambda(get_reranked_docs) | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | RunnableLambda(llm_chain_call)
        | StrOutputParser()
    )

    # Execute
    top_docs = get_reranked_docs(question)
    answer = rag_chain.invoke(question)
    
    return answer, top_docs