# Set Up Environment
* `git clone https://github.com/yunfanY-bot/starward_ai.git`
* `pip install -r requirements.txt`

# Model 

graph TD
    A[Input query] --> B
    Z[Chat history] --> B
    subgraph history_aware_retriever
    B["Contextualize query" prompt] --> C[LLM]
    C --> D[Contextualized query]
    D --> E[Retriever]
    E --> F[Documents]
    end
    subgraph question_answer_chain
    F --> G["Answer question" prompt]
    A --> G
    Z --> G
    G --> H[LLM]
    H --> I[Answer]
    end