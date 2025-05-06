from langsmith import Langsmith
from qa_chain import get_answer
from langchain.chains import ConversationalRetrievalChain

def test_performance(user_question):
    """
    Test model performance using the user's input question and check for hallucination.
    """
    client = Langsmith(api_key="your_langsmith_api_key")
    
    if not user_question.strip():
        return "⚠️ Please enter a question before testing performance."
    
    try:
        # Get the answer from the model
        model_answer = get_answer(user_question)
        
        # Evaluate the answer using LangSmith
        result = client.evaluate(model_answer)
        
        # Check if hallucination is present in the result
        if "hallucination" in result.lower():  # This is a conceptual check; adjust based on actual LangSmith output
            result += "\n⚠️ Hallucination detected in the answer."
        return result
    except Exception as e:
        return f"❌ Error evaluating the model: {e}"

# Example manual input
user_question = input("🔹 Enter your question to test the model: ")
performance_result = test_performance(user_question)

# Print question and performance result
print(f"\n📝 Question: {user_question}\n✅ Result: {performance_result}")
