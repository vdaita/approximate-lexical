import al_rust
import bm25s
import typer
import Stemmer
from bm25s.scoring import _calculate_doc_freqs, _build_idf_array, _select_idf_scorer
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

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
            "What is the diffusion coefficient of cerebral white matter?",
            "Is chemotherapy effective for treating cancer?",
            "Is cardiac injury common in critical cases of COVID-19?",
            "How is TLR7 stimulated in plasmacytoid cells?",
            "Where are chromatin histones elongated?"
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

    approx_index = al_rust.build_approx_index(
        scored_documents=retriever.scores,
        vocab_size=vocab_size
    )

    bm25s_results = []
    approx_results = []
    coverage_topks = []

    for query in tqdm(queries, desc="Processing queries"):
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, leave=False)
        
        # BM25s search
        bm25s_retrieved_documents = retriever.retrieve(query_tokens, k=top_k_target)
        bm25s_retrieved_documents_ids = bm25s_retrieved_documents[0][0]
        bm25s_results.append(bm25s_retrieved_documents_ids)

        # Prepare weighted query: (token_id, token_weight)
        query_tokens = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)
        query_token_ids = retriever.get_tokens_ids(query_tokens)
        query_token_weights = [token_weights[token_id] for token_id in query_token_ids]
        weighted_query = list(zip(query_token_ids, query_token_weights))

        # Find minimal top_k for approx search to cover all bm25 top_k
        found_k = None
        for k in range(top_k_target, corpus_size, max(10, top_k_target)):
            approx_top_ids = al_rust.search_approx_index(
                index=approx_index,
                query=weighted_query,
                top_k=k
            )
            if bm25s_retrieved_documents_ids.issubset(approx_top_ids):
                found_k = k
                break
        if found_k is None:
            found_k = corpus_size
        coverage_topks.append(found_k)
        approx_results.append(approx_top_ids)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(queries)), coverage_topks, marker='o')
    plt.xlabel('Query Index')
    plt.ylabel('Minimal top_k for full coverage')
    plt.title(f'Minimal top_k needed for Approximate Search to cover BM25s top {top_k_target}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    typer.run(main)