from transformers import AutoTokenizer, RagRetriever, RagModel
from retriever import get_retriever
import torch

def predict():
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = get_retriever()
    # initialize with RagRetriever to do everything in one forward call
    model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

    inputs = tokenizer("How long is the AIAP program?", return_tensors="pt")
    outputs = model(input_ids=inputs["input_ids"])

    generated_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_string