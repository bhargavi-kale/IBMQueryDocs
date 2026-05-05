"""RAG pipeline module for processing queries with retrieval and reranking."""

# def run_rag_pipeline(question, retriever, rerank_model, model, tokenizer):
#     initial_docs = retriever.invoke(question)
#     if not initial_docs:
#         return "No relevant documents found.", []

#     pairs = [[question, doc.page_content] for doc in initial_docs]
#     scores = rerank_model.predict(pairs)
#     scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0],
#                          reverse=True)
#     top_docs = [doc for _, doc in scored_docs[:3]]

#     context = "\n\n".join([doc.page_content for doc in top_docs])

#     # MATCHING NOTEBOOK LOGIC (Section 7)
#     messages = [
#         {
#             "role": "system",
#             "content": """You are a strict RAG system.
#             You ONLY use explicitly stated context. No external knowledge.""",
#         },
#         {
#             "role": "user",
#             "content": f"""
#         Step 1: Extract exact relevant lines from context.
#         Step 2: If none exist, say "Insufficient context."
#         Step 3: Answer ONLY using extracted lines, be concise and factual.
#         Step 4: Do not combine multiple components into a higher-level product
#         definition unless explicitly stated in context.

#         Context:
#         {context}

#         Question:
#         {question}
#         """,
#         },
#     ]

#     # Notebook fix: return_dict=True is required for proper unpacking
#     inputs = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt",
#         return_dict=True
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=400,
#             temperature=0.1,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     prompt_length = inputs["input_ids"].shape[-1]
#     answer = tokenizer.decode(outputs[0][prompt_length:],
#                               skip_special_tokens=True)

#     return answer, top_docs


def run_rag_pipeline(question, retriever, rerank_model, llm):
    """
    Run a RAG pipeline: retrieve, rerank, and generate an answer.

    Retrieves documents for the question, reranks them, builds a context
    with source tags, and queries the LLM for a grounded response.

    Args:
        question (str): User query.
        retriever: Document retriever with `.invoke()`.
        rerank_model: Model to score query-document pairs.
        llm: Language model with `.invoke()`.

    Returns:
        tuple: (answer (str), top_docs (list))
    """
    # 1. Retrieval & Reranking logic stays the same[cite: 6]
    initial_docs = retriever.invoke(question)
    if not initial_docs:
        return "No relevant documents found.", []

    pairs = [[question, doc.page_content] for doc in initial_docs]
    scores = rerank_model.predict(pairs)
    scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:3]]

    context_parts = []
    for doc in top_docs:
        source_id = doc.metadata.get("source", "Unknown")
        # Prepend the source ID to the chunk so the LLM knows where it came from
        context_parts.append(f"[Source: {source_id}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)
    # 2. Simplified Prompting
    # LangChain's HuggingFaceEndpoint handles the formatting
    prompt = [
        {
            "role": "system",
            "content": """"You are a strict RAG system. You ONLY use explicitly
            stated context. No external knowledge. You MUST cite your sources
            for every factual claim you make using the provided [Source: ID]
            tags.""",
        },
        {
            "role": "user",
            "content": f"""
        Step 1: Extract exact relevant lines from context.
        Step 2: If none exist, say "Insufficient context."
        Step 3: Answer ONLY using extracted lines, be concise and factual.
        Step 4: Only if an answer is found, append the corresponding source ID. If you say 'Insufficient context', do not append a source ID.
        Step 5: Do not combine multiple components into a higher-level product definition unless explicitly stated in context.
        Context:
        {context}

        Question:
        {question}
        """,
        },
    ]

    # 3. Direct Invocation
    response = llm.invoke(prompt)
    answer = response.content

    return answer, top_docs
