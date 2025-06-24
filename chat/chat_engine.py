from retrieval.hybrid_retriever import HybridRetriever
from llm_handler import generate_response
class ChatEngine:
    def __init__(self, memory):
        self.memory = memory
        self.retriever = HybridRetriever

    def handle_query(self, query):
        """
        Handle a user query by retrieving relevant documents and generating a response.
        Args:
            query (str): The user's query.
        Returns:
            str: The generated response.
        """
        self.memory.append("user", query)
        # Retrieve relevant documents using hybrid_retriever
        docs = self.retriever.retrieve(query, top_k=5)
        context = "\n".join([doc['content'] for doc in docs])
        answer = generate_response(context, query)
        self.memory.append("assistant", answer)
        return answer

    def generate_response(self, query, intent, sentiment, context, initial_response=None):
        """
        Generate a response based on query, intent, sentiment, and context.
        Args:
            query (str): The user's query.
            intent (str): The classified intent of the query.
            sentiment (str): The sentiment of the query.
            context (dict): The conversation context.
            initial_response (str, optional): An initial response to refine.
        Returns:
            str: The final response.
        """
        if initial_response:
            return initial_response  # Use the initial response if provided
        docs = self.retriever.retrieve(query, top_k=5)
        retrieved_context = "\n".join([doc['content'] for doc in docs])
        full_context = f"{context.get('history', '')}\n{retrieved_context}"
        return generate_response(full_context, query)