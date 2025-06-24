from models.lstm_generator import lstm_generator
from models.transformer_generator import generate_response_transformer

def route_query(query, intent, retriever, models):
    context = None

    # Try retrieving relevant context (if retriever is available)
    if retriever:
        try:
            results = retriever.retrieve(query, top_k=1)
            if results:
                context = results[0][0]  # top chunk text
        except Exception as e:
            context = None

    # Use model depending on intent
    if intent == "general":
        transformer_model = models.get("transformer_model")
        stoi = models.get("transformer_stoi")
        itos = models.get("transformer_itos")
        response = generate_response_transformer(query, transformer_model, stoi, itos, context=context)
    else:
        lstm_model = models.get("lstm_model")
        stoi = models.get("lstm_stoi")
        itos = models.get("lstm_itos")
        response = lstm_generator(query, lstm_model, stoi, itos, context=context)

    return response, context


def handle_query_response(query, intent, sentiment, generator, retriever, lstm_model=None, lstm_stoi=None, lstm_itos=None):
    """
    Handle query response generation based on intent and sentiment.
    Args:
        query (str): The input query.
        intent (str): The detected intent of the query.
        sentiment (str): The detected sentiment of the query.
        generator: The generator object (e.g., transformer generator).
        retriever: The retriever object for RAG-based generation.
        lstm_model: The LSTM model for specific intents.
        lstm_stoi (dict): String-to-index vocabulary mapping for LSTM.
        lstm_itos (dict): Index-to-string vocabulary mapping for LSTM.
    Returns:
        tuple: (response, context) - The generated response and retrieved context.
    """
    models = {
        "lstm_model": lstm_model,
        "lstm_stoi": lstm_stoi,
        "lstm_itos": lstm_itos,
        "transformer_model": generator,
        "transformer_stoi": lstm_stoi,   # reuse vocab if applicable
        "transformer_itos": lstm_itos
    }

    response, context = route_query(query, intent, retriever, models)
    
    # Adjust response based on sentiment (simplified example)
    if sentiment == "negative":
        response = f"I'm sorry to hear that. {response}"
    
    return response, context