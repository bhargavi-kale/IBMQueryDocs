## Sprint1

## 1. What project work have you achieved since last class last week?

- Loaded the `google/flan-T5-base` model in the project.
- Used similarity search to retrieve the top 5 chunks.
- Created a prompt to ensure the model answers based on the context and returns “I don’t know” when it does not have sufficient information.
- Built a RAG pipeline to generate answers for queries.


## 2. What challenges, setbacks, or blockers have you encountered?

- The model is generating answers from the context, but it is not properly answering the question.
- Sometimes it returns “I don’t know” even when relevant information is present.


## 3. What is your actionable and feasible plan for this week’s sprint?

### a. Time allocation and schedule
- Thursday: 3–4 hours  
- Sunday: 3–4 hours  

### b. Specific activities/tasks
- Debug the pipeline by printing each step implemented so far.
- Validate each stage and identify issues.
- Explore other approaches to improve answer quality.

### c. Tangible work products
- Updates to the implementation of query fetching and the RAG pipeline.

## Sprint2

## 1. What project work have you achieved since last class last week?
- Created a separate jupyter notebook to test and debug the issues.
- Tested the `google/flan-T5-base` model for descriptive answers by passing an entire document and asking it to describe it in 250 words. It returned only 4–5 words, so I changed the model to Mistral `(mistralai/Mistral-7B-Instruct-v0.2)`.
- Tested the retrieval accuracy using the dataset and checked whether the original source ID is present in the retrieved chunks. I changed the chunk size to 1000 and the chunk overlap to 300, which increased my accuracy to 90%.
- Performed reranking of chunks using `BAAI/bge-reranker-base.`


## 2. What challenges, setbacks, or blockers have you encountered?

- Got very short answers from the LLM.
- Retrieved chunks do not have content related to the question.
- The answer is not complete because the remaining part is in a different chunk.


## 3. What is your actionable and feasible plan for this week’s sprint?

### a. Time allocation and schedule
- Thursday: 3–4 hours  
- Saturday: 3–4 hours  

### b. Specific activities/tasks
- Planning to pass the entire document to the LLM based on the top 3 chunks for better and complete answer.

### c. Tangible work products
- Changes in the RAG pipeline method and corrections to it.
- Integrate the changes from my test file(mistrall_test) to main.ipynb.

## Sprint3

## 1. What project work have you achieved since last class last week?
- Attempted testing the application using the `RAGAS` framework.
- Modularized the Jupyter Notebook into Python modules within the src folder.
- Worked on project documentation (README.md).


## 2. What challenges, setbacks, or blockers have you encountered?

- Faced timeout errors while running the RAGAS framework; execution often failed even after running for nearly an hour.
- Encountered issues connecting to the kernel after converting the notebook into Python modules.


## 3. What is your actionable and feasible plan for this week’s sprint?

### a. Time allocation and schedule
- Thursday: 3–4 hours  
- Saturday: 3–4 hours  

### b. Specific activities/tasks
- Identify ways to execute Python files similarly to Jupyter Notebook workflows.
- Implement citation support for generated answers.
- Work on deployment using Hugging Face and Gradio.

### c. Tangible work products
- Successfully run the complete RAG pipeline using Python files.
- Include source IDs along with generated answers.
- Deploy the application on Hugging Face and Gradio.