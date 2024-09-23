# chatbot.py
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering # type: ignore
import torch # type: ignore
import random

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Sample context
context = """
Our customer support is available 24/7. You can reach us via email at support@example.com or call us at 123-456-7890.
We provide a variety of products including electronics, clothing, and home goods. 
We also offer shipping options, returns, and payment methods including credit cards and PayPal.
"""

greetings = ["Hello!", "Hi there!", "Greetings!", "How can I assist you today?"]

# Dictionary of specific topics
topic_responses = {
    "support": "You can reach our support team at support@example.com or call us at 123-456-7890.",
    "products": "We offer a variety of products including electronics, clothing, and home goods.",
    "shipping": "We provide standard and expedited shipping options.",
    "returns": "You can return products within 30 days of purchase.",
    "payment": "We accept credit cards and PayPal for your convenience."
}

def answer_question(question):
    # Check for greetings
    if any(greeting.lower() in question.lower() for greeting in ["hello", "hi", "greetings"]):
        return random.choice(greetings)

    # Check for specific topics
    for topic, response in topic_responses.items():
        if topic in question.lower():
            return response

    # Use the QA model for other questions
    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax() + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer or "I'm sorry, I don't understand the question."
