from transformers import AutoTokenizer, RagModel
from retriever import get_retriever
from transformers import RagTokenizer, RagSequenceForGeneration
import torch


def predict(question):
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = get_retriever()
    # initialize with RagRetriever to do everything in one forward call
    model = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq", retriever=retriever
    )
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    return generated_string


if __name__ == "__main__":
    question = input("Question about AIAP: ")
    answer = predict(question)
    print(f"Answer: {answer}")
