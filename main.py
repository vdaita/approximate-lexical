import matplotlib

# Try to use TkAgg for interactive, fallback to Agg for headless environments
try:
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    INTERACTIVE = True
except Exception:
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    INTERACTIVE = False

import al_rust
import bm25s
import typer
import Stemmer
from bm25s.scoring import _calculate_doc_freqs, _build_idf_array, _select_idf_scorer
from tqdm import tqdm
from rich import print
import numpy as np
import scipy.sparse as sp

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

def main(source: str, top_k_target: int):
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

    corpus_ids = list(range(len(corpus_lst)))
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
    for i,j,v in tqdm(zip(scores_coo.row, scores_coo.col, scores_coo.data)):
        posting_list[j].append((i, v))

    approx_index = al_rust.build_approx_index(
        scored_documents=posting_list,
        vocab_size=vocab_size
    )

    bm25s_results = []
    approx_results = []

    coverage_topks = []
    upper_bound = [] # compare to the sum of lengths of posting lists

    for query in tqdm(queries, desc="Processing queries"):
        
        # BM25s search
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, leave=False)
        bm25s_retrieved_documents = retriever.retrieve(query_tokens, k=top_k_target)
        bm25s_retrieved_documents_ids = bm25s_retrieved_documents[0][0]
        bm25s_results.append(bm25s_retrieved_documents_ids)

        weighted_query, num_tokens = embed_query(query, retriever, token_weights)

        # Find minimal top_k for approx search to cover all bm25 top_k
        found_k = None
        for k in range(top_k_target, corpus_size, max(10, top_k_target)):
            approx_top_ids = al_rust.search_approx_index(
                index=approx_index,
                query=weighted_query,
                top_k=k
            )
            if set(bm25s_retrieved_documents_ids).issubset(approx_top_ids):
                found_k = k
                break
        if found_k is None:
            found_k = corpus_size

        upper_bound_sum = sum(len(posting_list[token]) for token in [pair[0] for pair in weighted_query])
        print("Tokens that have 0 coverage:", [pair[0] for pair in weighted_query if len(posting_list[pair[0]]) == 0], " out of tokens ", [pair[0] for pair in weighted_query])
        coverage_topks.append(found_k / (1 + upper_bound_sum))
        approx_results.append(approx_top_ids)

    # Print recall results for each query
    for i, query in enumerate(queries):
        bm25_ids = set(bm25s_results[i])
        recalls = []
        weighted_query, _ = embed_query(query, retriever, token_weights)
        query_token_ids = [pair[0] for pair in weighted_query]
        total_posting_list_elements = sum(len(posting_list[token_id]) for token_id in query_token_ids)

        ks = list(range(top_k_target, int(coverage_topks[i]*corpus_size)+1, max(1, top_k_target//5)))
        fractions = []
        for k in ks:
            posting_list_elements_accessed = total_posting_list_elements
            fraction_accessed = posting_list_elements_accessed / (1 + total_posting_list_elements)
            fractions.append(fraction_accessed)

        for idx, k in enumerate(ks):
            approx_ids = al_rust.search_approx_index(
                index=approx_index,
                query=weighted_query,
                top_k=k
            )
            recall = len(bm25_ids.intersection(approx_ids)) / len(bm25_ids)
            recalls.append(recall)
        print(f"\n[bold blue]Query {i+1}:[/bold blue] {query}")
        for frac, rec, k in zip(fractions, recalls, ks):
            print(f"  Top-k={k}, Fraction Viewed={frac:.4f}, Recall={rec:.4f}")

if __name__ == "__main__":
    typer.run(main)