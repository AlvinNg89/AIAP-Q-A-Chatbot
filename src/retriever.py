# To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
from transformers import RagRetriever


def get_retriever():
    dataset = (
        ...
    )  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index

    retriever = RagRetriever.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base", indexed_dataset=dataset
    )

    # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py

    dataset_path = "path/to/my/dataset"  # dataset saved via *dataset.save_to_disk(...)*

    index_path = "path/to/my/index.faiss"  # faiss index saved via *dataset.get_index("embeddings").save(...)*

    retriever = RagRetriever.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path,
    )

    return retriever
