# To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
from transformers import RagRetriever
from datasets import load_from_disk


def get_retriever():
    dataset_path = "./training_data/my_knowledge_dataset"
    index_path = "./training_data/my_knowledge_dataset_hnsw_index.faiss"
    # dataset = load_from_disk(dataset_path)
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path,
    )

    # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py

    # dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*

    # index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*

    # retriever = RagRetriever.from_pretrained(
    #     "facebook/dpr-ctx_encoder-single-nq-base",
    #     index_name="custom",
    #     passages_path=dataset_path,
    #     index_path=index_path,
    # )

    return retriever
