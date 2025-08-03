import matplotlib

import al_rust
import bm25s
import typer
import Stemmer
from bm25s.scoring import _calculate_doc_freqs, _build_idf_array, _select_idf_scorer
from tqdm import tqdm
from rich import print
import numpy as np
from rich.table import Table
from rich.console import Console
import scipy.sparse as sp
import matplotlib.pyplot as plt

def embed_query(query: str, retriever: bm25s.BM25, token_weights: np.ndarray):
    """
    Embed the query into a weighted token representation.
    """
    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=Stemmer.Stemmer("english"), leave=False)
    query_tokens_str = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)[0]
    query_token_ids = retriever.get_tokens_ids(query_tokens_str)
    query_token_ids_count = np.bincount(query_token_ids, minlength=len(token_weights))
    normalized_weights = query_token_ids_count / np.sum(query_token_ids_count)
    unique_token_ids = np.unique(query_token_ids)
    pairs = [(token_id, normalized_weights[token_id]) for token_id in unique_token_ids]
    return pairs, np.sum(query_token_ids_count)

def main(source: str, top_k_target: int, micro: bool):
    if source == "scifact":
        queries = [
            "what is the diffusion coefficient of cerebral white matter?",
            "Is chemotherapy effective for treating cancer?",
            "Is Cardiac injury is common in critical cases of COVID-19?",
            "how is tlr7 stimulated in plasmacytoid cells",
            "where is chromatin histones elongated"
        ]
    elif source == "quora":
        queries = [
            "What is the story of Kohinoor (Koh-i-Noor) Diamond?",
            "Does Quora have a character limit for profile descriptions?",
            "Is Career Launcher good for RBI Grade B preparation?",
            "Is 7 days too late for rabies vaccine after a possible non-bite exposure?",
            "Is atheism the lack of belief in gods, or a claim that god does not exist?"
        ]
    else:
        raise ValueError("Source must be either 'scifact' or 'quora'")

    dataset_dir = f"data/{source}"
    bm25s.utils.beir.download_dataset(dataset=source, save_dir=dataset_dir)
    corpus = bm25s.utils.beir.load_corpus(dataset=source, save_dir=dataset_dir)

    corpus_lst = []
    for key, val in corpus.items():
        corpus_lst.append(val["title"] + " " + val["text"])
    del corpus

    corpus_lst = list(corpus_lst)
    corpus_size = len(corpus_lst)

    stemmer = Stemmer.Stemmer("english")
    
    corpus_tokens = bm25s.tokenize(
        corpus_lst, stemmer=stemmer, leave=False
    )
    vocab_size = len(corpus_tokens.vocab)

    token_weights = _calculate_doc_freqs(corpus_tokens.ids, unique_tokens=list(range(vocab_size)))
    token_weights = _build_idf_array(token_weights, corpus_size, compute_idf_fn=_select_idf_scorer('lucene'))

    retriever = bm25s.BM25(backend='numba')
    retriever.index(corpus_tokens)

    scores_coo = sp.coo_matrix(
        sp.csc_matrix(
            (retriever.scores['data'], retriever.scores['indices'], retriever.scores['indptr'])
        )
    )

    posting_list = [[] for _ in range(vocab_size)]

    print("Maximum column: ", max(scores_coo.col))
    print("Maximum row: ", max(scores_coo.row))
    print("Number of non-zero entries: ", scores_coo.nnz)

    for i,j,v in tqdm(zip(scores_coo.row, scores_coo.col, scores_coo.data)):
        posting_list[j].append((i, v))

    approx_index = al_rust.build_approx_index(
        scored_documents=posting_list,
        vocab_size=vocab_size,
        micro=micro
    )

    bm25s_results = []
    points = []

    if micro:
        print("Microbenchmark mode enabled. This will only process the first query.")
        query = queries[0]
        top_k_target = 32  # Set a small top_k for microbenchmarking

        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, leave=False)
        bm25s_retrieved_documents = retriever.retrieve(query_tokens, k=top_k_target)
        bm25s_retrieved_documents_ids = bm25s_retrieved_documents[0][0]

        print("BM25s Retrieved Documents IDs:", bm25s_retrieved_documents_ids)
        weighted_query, num_tokens = embed_query(query, retriever, token_weights)

        approx_top_ids = al_rust.search_approx_index(
            index=approx_index,
            query=weighted_query,
            top_k=top_k_target,
            micro=micro
        )
        intersection_amount = len(set(bm25s_retrieved_documents_ids).intersection(approx_top_ids))

        print("Query:", query)
        print("Approx Top IDs:", approx_top_ids)
        print("Intersection size: ", intersection_amount)
        
        return

    for query_index, query in tqdm(enumerate(queries), desc="Processing queries"):
        
        # BM25s search
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, leave=False)
        bm25s_retrieved_documents = retriever.retrieve(query_tokens, k=top_k_target)
        bm25s_retrieved_documents_ids = bm25s_retrieved_documents[0][0]
        bm25s_results.append(bm25s_retrieved_documents_ids)

        weighted_query, num_tokens = embed_query(query, retriever, token_weights)

        # Find minimal top_k for approx search to cover all bm25 top_k
        for k in range(top_k_target, corpus_size, max(10, top_k_target)):
            approx_top_ids = al_rust.search_approx_index(
                index=approx_index,
                query=weighted_query,
                top_k=k,
                micro=micro
            )
            intersection_amount = len(set(bm25s_retrieved_documents_ids).intersection(approx_top_ids))
            if k == top_k_target: # this is the first one:
                print(f"Query {query_index+1}: BM25s top-k={len(bm25s_retrieved_documents_ids)}, Approx top-k={k}, Intersection={intersection_amount}, Retrieved documents={approx_top_ids}")

            proportion_seen = k / top_k_target
            coverage = intersection_amount / top_k_target
            points.append(
                (query_index, proportion_seen, coverage)
            )

    # Convert points to numpy array for easier plotting
    points_arr = np.array(points)

    # Plot each query's points with a different color
    for query_index in range(len(queries)):
        query_points = points_arr[points_arr[:,0] == query_index]
        plt.plot(query_points[:,1], query_points[:,2], label=f"Query {query_index+1}")

        # Report out in text with rich
        table = Table(title=f"Query {query_index+1}: {queries[query_index]}")
        table.add_column("Proportion Seen", justify="right")
        table.add_column("Coverage", justify="right")
        for x, y in zip(query_points[:,1], query_points[:,2]):
            table.add_row(f"{x:.3f}", f"{y:.3f}")
        console = Console()
        console.print(table)

    plt.xlabel("Proportion Seen")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Proportion Seen for Each Query")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    typer.run(main)