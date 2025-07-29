import bm25s
import Stemmer
import time
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List
import functools
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from bm25s.numba.retrieve_utils import _retrieve_numba_functional
from bm25s.scoring import _calculate_doc_freqs, _build_idf_array, _select_idf_scorer
import scipy
from tqdm import tqdm
import numpy as np
import hnswlib

def generate_random_matrix(dim: int = 256, num_tokens: int = 25600):
    mat = np.random.randn(num_tokens, dim).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms

def main():
    dataset = "scifact"
    dataset_dir = f"data/{dataset}"

    queries = [
        "what is the diffusion coefficient of cerebral white matter?"
    ]

    # queries = [
    #     "What is the story of Kohinoor (Koh-i-Noor) Diamond?",
    #     "Does Quora have a character limit for profile descriptions?",
    #     "Is Career Launcher good for RBI Grade B preparation?",
    #     "Is 7 days too late for rabies vaccine after a possible non-bite exposure?",
    #     "Is atheism the lack of belief in gods, or a claim that god does not exist?"
    # ]

    bm25s.utils.beir.download_dataset(dataset=dataset, save_dir=dataset_dir)
    corpus = bm25s.utils.beir.load_corpus(dataset=dataset, save_dir=dataset_dir)
    
    corpus_lst = []
    for key, val in corpus.items():
        corpus_lst.append(val["title"] + " " + val["text"])
    del corpus

    corpus_ids = list(range(len(corpus_lst)))
    corpus_lst = list(corpus_lst)
    corpus_size = len(corpus_lst)

    print("IDs list: ", corpus_ids)

    stemmer = Stemmer.Stemmer("english")
    
    corpus_tokens = bm25s.tokenize(
        corpus_lst, stemmer=stemmer, leave=False
    )
    print("Corpus tokens: ", corpus_tokens)
    vocab_size = len(corpus_tokens.vocab)
    print("Vocabulary size: ", vocab_size)
    token_weights = _calculate_doc_freqs(corpus_tokens.ids, unique_tokens=list(range(vocab_size)))
    token_weights = _build_idf_array(token_weights, corpus_size, compute_idf_fn=_select_idf_scorer('lucene'))

    print("Document frequencies: ", token_weights, " Token weights shape: ", token_weights.shape)

    retriever = bm25s.BM25(backend='numba')
    retriever.index(corpus_tokens)
    top_k = 8
    dim_size = 512

    random_matrix = generate_random_matrix(dim=dim_size, num_tokens=vocab_size)
    corpus_embeddings = np.zeros((corpus_size, dim_size), dtype=np.float32)
    for doc_id, doc in tqdm(enumerate(corpus_tokens.ids), total=corpus_size, desc="Processing corpus"):
        for token in doc:
            corpus_embeddings[doc_id] += random_matrix[token] * token_weights[token]
        corpus_embeddings[doc_id] /= np.linalg.norm(corpus_embeddings[doc_id]) if np.linalg.norm(corpus_embeddings[doc_id]) > 0 else 1 
        
    index = hnswlib.Index(space='l2', dim=dim_size)
    index.init_index(max_elements=len(corpus_lst), ef_construction=200, M=16)
    index.add_items(corpus_embeddings, list(range(len(corpus_lst))))
    index.set_ef(top_k * 2)

    for query in tqdm(queries, desc="Processing queries"):
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        baseline_retrieved_documents = retriever.retrieve(query_tokens, k=top_k)
        baseline_retrieved_document_ids = baseline_retrieved_documents[0][0]
        print("Baseline retrieved documents: ", baseline_retrieved_documents)

        query_tokens = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)[0]
        print("Query tokens: ", query_tokens)

        query_ids = retriever.get_tokens_ids(query_tokens)

        query_vector = np.zeros((dim_size,), dtype=np.float32)
        for token in query_ids:
            query_vector += random_matrix[token] * token_weights[token]

        approx_labels, approx_distances = index.knn_query(query_vector, k=top_k)
        approx_labels, approx_distances = approx_labels[0], approx_distances[0]

        print("Approx labels: ", approx_labels)
        print("Approx distances: ", approx_distances)

        document_intersection = set(baseline_retrieved_document_ids).intersection(set(approx_labels))
        print("Intersection of retrieved documents: ", document_intersection)

        print("Intersection ratio: ", len(document_intersection), " / ", top_k)

        # approximate retrieval

if __name__ == "__main__":
    main()