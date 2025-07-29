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
from scipy.linalg import hadamard

def generate_random_matrix(dim: int = 256, num_tokens: int = 25600):
    mat = np.random.randn(num_tokens, dim).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms

def report_approximate_distances(random_matrix, n=10000):
    distances = []
    for _ in range(n):
        vec_index_1 = np.random.randint(0, random_matrix.shape[0])
        vec_index_2 = np.random.randint(0, random_matrix.shape[0])
        while vec_index_1 == vec_index_2:
            vec_index_2 = np.random.randint(0, random_matrix.shape[0])
        vec_1 = random_matrix[vec_index_1]
        vec_2 = random_matrix[vec_index_2]
        distances.append(np.dot(vec_1, vec_2))
    
    distances = np.array(distances)
    print(f"Mean dot product: {distances.mean():.4f}")
    print(f"Std deviation: {distances.std():.4f}")
    print(f"Min dot product: {distances.min():.4f}")
    print(f"Max dot product: {distances.max():.4f}")
    print(f"25th percentile: {np.percentile(distances, 25):.4f}")
    print(f"50th percentile (median): {np.percentile(distances, 50):.4f}")
    print(f"75th percentile: {np.percentile(distances, 75):.4f}")

def main():
    dataset = "scifact"
    # dataset = "quora"
    dataset_dir = f"data/{dataset}"

    queries = [
        "what is the diffusion coefficient of cerebral white matter?",
        "Is chemotherapy effective for treating cancer?",
        "Is Cardiac injury is common in critical cases of COVID-19?",
        "how is tlr7 stimulated in plasmacytoid cells",
        "where is chromatin histones elongated"
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

    print("Corpus size: ", corpus_size)

    # print("IDs list: ", corpus_ids)

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
    top_k = 32
    dim_size = 1024

    k1 = retriever.k1
    b = retriever.b

    top_k_multiplier = 10

    random_matrix = generate_random_matrix(dim=dim_size, num_tokens=vocab_size)

    print("Random matrix orthogonality check: ")
    report_approximate_distances(random_matrix)

    corpus_embeddings = np.zeros((corpus_size, dim_size), dtype=np.float32)
    doc_lengths = np.array([len(doc) for doc in corpus_tokens.ids])
    average_doc_length = np.mean(doc_lengths)

    print("Average document length: ", average_doc_length)

    for doc_id, (doc, doc_length) in tqdm(enumerate(zip(corpus_tokens.ids, doc_lengths)), total=corpus_size, desc="Processing corpus"):
        token_counts = np.bincount(doc, minlength=vocab_size)
        for token, count in enumerate(token_counts):
            if count > 0:
                doc_token_score = (token_counts[token] * (k1 + 1)) / (token_counts[token] * (k1 * (1 - b + b * doc_length / average_doc_length)))
                # corpus_embeddings[doc_id] += (random_matrix[token]) * (token_counts[token] * (k1 + 1)) / (token_counts[token] * (k1 * (1 - b + b * doc_length / average_doc_length)))
                # corpus_embeddings[doc_id] += (random_matrix[token]) * (token_counts[token])

                corpus_embeddings[doc_id] += random_matrix[token] * doc_token_score * token_weights[token]
        norm = np.linalg.norm(corpus_embeddings[doc_id])
        corpus_embeddings[doc_id] /= norm if norm > 0 else 1

    index = hnswlib.Index(space='l2', dim=dim_size)
    index.init_index(max_elements=len(corpus_lst), ef_construction=50, M=32)
    index.add_items(corpus_embeddings, list(range(len(corpus_lst))))
    index.set_ef(400)

    print("Finished building HNSW index.")

    for query in tqdm(queries, desc="Processing queries"):
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        baseline_retrieved_documents = retriever.retrieve(query_tokens, k=top_k)
        baseline_retrieved_document_ids = baseline_retrieved_documents[0][0]
        print("Baseline retrieved documents: ", baseline_retrieved_documents)

        query_tokens = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)[0]
        print("Query tokens: ", query_tokens)

        query_ids = retriever.get_tokens_ids(query_tokens)

        query_vector = np.zeros((dim_size,), dtype=np.float32)
        query_token_counts = np.bincount(query_ids, minlength=vocab_size)
        for token, count in enumerate(query_token_counts):
            if count > 0:
                query_vector += random_matrix[token] * count

        query_vector /= np.linalg.norm(query_vector) if np.linalg.norm(query_vector) > 0 else 1

        approx_labels, approx_distances = index.knn_query(query_vector, k=top_k * top_k_multiplier)
        approx_labels, approx_distances = approx_labels[0], approx_distances[0]

        real_approx_labels = approx_labels[:top_k]

        print("Approx labels: ", approx_labels)
        print("Approx distances: ", approx_distances)

        document_intersection = set(baseline_retrieved_document_ids).intersection(set(approx_labels))
        print("Intersection of retrieved documents: ", document_intersection)
        print("Intersection ratio with first stage: ", len(document_intersection), " / ", top_k)

        # Figure out what happens when you use the NN as first stage retrieval:
        real_document_intersection = set(baseline_retrieved_document_ids).intersection(set(real_approx_labels))
        print("Real intersection of retrieved documents: ", real_document_intersection)
        print("Real intersection ratio: ", len(real_document_intersection), " / ", top_k)

        # approximate retrieval

if __name__ == "__main__":
    main()