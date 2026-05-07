from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def run_rag_pipeline(question, retriever, rerank_model, client):
    
    # --- Step A: Single Retrieval & Reranking Logic ---
    def get_reranked_docs(query):
        initial_docs = retriever.invoke(query)
        if not initial_docs: 
            return []
        
        # Prepare pairs for reranker
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = rerank_model.predict(pairs)
        
        # Sort and take top 3
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:3]]

    # --- Step B: Enhanced Formatting ---
    def format_docs(docs):
        """Pairs ID and Content tightly so the model can't miss them."""
        formatted_chunks = []
        for d in docs:
            sid = d.metadata.get('source', 'N/A')
            # Clean content of newlines to keep it as a single block
            content = d.page_content.replace('\n', ' ')
            formatted_chunks.append(f"SOURCE_ID: {sid}\nCONTENT: {content}")
        return "\n\n---\n\n".join(formatted_chunks)

    # --- Step C: High-Instruction Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise technical assistant for IBM documentation.
        
        RULES:
        1. Answer ONLY using the provided DOCUMENTS.
        2. You MUST append the Source ID at the end of every sentence, like this: [Source: ID].
        3. If the answer is not in the documents, say: 'The provided documentation does not contain information about this.'
        4. Do NOT use your own knowledge or mention anything not in the text below."""),
                ("user", "DOCUMENTS:\n{context}\n\nQUESTION: {question}")
    ])

    # --- Step D: LLM Call ---
    def llm_chain_call(prompt_input):
        # prompt_input is the formatted prompt from the chain
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt_input.to_string()}],
            max_tokens=600,
            temperature=0  # Zero temperature is critical to stop hallucinations
        )
        return response.choices[0].message.content

    # --- Step E: Execution ---
    # We fetch docs once to avoid redundant work and inconsistency
    top_docs = get_reranked_docs(question)
    context_text = format_docs(top_docs)
    
    # Run the chain manually for better control
    chain_input = {"context": context_text, "question": question}
    formatted_prompt = prompt.invoke(chain_input)
    answer = llm_chain_call(formatted_prompt)
    
    return answer, top_docs