import approximate_lexical as al
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

    print("Maximum column: ", max(scores_coo.col))
    print("Maximum row: ", max(scores_coo.row))
    print("Number of non-zero entries: ", scores_coo.nnz)

    approx_index = al.build_approx_index(
        corpus_tokens.ids,
        vocab_size,
        token_weights,
        micro
    )

    bm25s_results = []
    points = []

    if micro:
        queries = queries[:1]

    for query_index, query in tqdm(enumerate(queries), desc="Processing queries"):
        
        # BM25s search
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, leave=False)
        bm25s_retrieved_documents = retriever.retrieve(query_tokens, k=top_k_target)
        bm25s_retrieved_documents_ids = bm25s_retrieved_documents[0][0]
        bm25s_results.append(bm25s_retrieved_documents_ids)

        query_tokens_str = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)[0]
        query_token_ids = retriever.get_tokens_ids(query_tokens_str)

        # Find minimal top_k for approx search to cover all bm25 top_k
        for k in range(top_k_target, corpus_size, max(10, top_k_target)):
            approx_top_ids = al.search_approx_index(
                approx_index,
                query_token_ids,
                k,
                micro
            )
            intersection_amount = len(set(bm25s_retrieved_documents_ids).intersection(approx_top_ids))
            proportion_seen = k / top_k_target
            coverage = intersection_amount / top_k_target
            points.append(
                (query_index, proportion_seen, coverage)
            )

            if micro:
                print("Query:", query)
                print("Approx Top IDs:", np.array(approx_top_ids))
                print("BM25s Retrieved IDs:", bm25s_retrieved_documents_ids)
                print("Intersection size: ", intersection_amount)
                return

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