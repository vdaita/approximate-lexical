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
import scipy

def main():
    dataset = "scifact"
    dataset_dir = f"data/{dataset}"

    queries = [
        "What is bone marrow?",
    ]

    bm25s.utils.beir.download_dataset(dataset=dataset, save_dir=dataset_dir)
    corpus = bm25s.utils.beir.load_corpus(dataset=dataset, save_dir=dataset_dir)
    
    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])
    del corpus

    corpus_ids, corpus_lst = zip(*sorted(zip(corpus_ids, corpus_lst), key=lambda x: x[1]))
    corpus_ids = list(corpus_ids)
    corpus_lst = list(corpus_lst)

    stemmer = Stemmer.Stemmer("english")
    
    corpus_tokens = bm25s.tokenize(
        corpus_lst, stemmer=stemmer, leave=False
    )

    retriever = bm25s.BM25(backend='numba')
    retriever.index(corpus_tokens)

    print(retriever)
    print(retriever.scores)

    print("Data: ", retriever.scores['data'], "Shape: ", retriever.scores['data'].shape)
    print("Indices: ", retriever.scores['indices'], "Shape: ", retriever.scores['indices'].shape)
    print("Index pointer: ", retriever.scores['indptr'], "Shape: ", retriever.scores['indptr'].shape)

    tensor_scores = torch.sparse_csc_tensor(
        ccol_indices=retriever.scores['indptr'],
        row_indices=retriever.scores['indices'],
        values=retriever.scores['data']
    )
    tensor_scores = tensor_scores.to_sparse_csc()

    print("Tensor scores: ", tensor_scores, " Shape: ", tensor_scores.shape)

    chunk_size = 8
    num_rows = tensor_scores.shape[0]
    num_chunks = (num_rows + chunk_size - 1) // chunk_size

    chunks = []
    document_chunk_list = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_rows)
        chunk_tensors = []
        for item_idx in range(start_idx, end_idx):
            chunk_tensors.append(torch.select_copy(tensor_scores, 0, item_idx))
        chunk_tensors = torch.stack(chunk_tensors, dim=0)
        chunk_average = torch.sparse.sum(chunk_tensors, dim=0) / (end_idx - start_idx + 1)
        chunks.append(chunk_average)
        document_chunk_list.append(chunk_tensors)

    chunked_tensor = torch.stack(chunks, dim=0)
    chunked_tensor = chunked_tensor.to_sparse_csc()

    chunked_tensor_coo = chunked_tensor.to_sparse_coo().coalesce()
    chunked_tensor_scipy = coo_matrix(
        (chunked_tensor_coo.values().numpy(), 
         (chunked_tensor_coo.indices()[0].numpy(), chunked_tensor_coo.indices()[1].numpy())),
        shape=chunked_tensor_coo.shape
    )
    chunked_tensor_scipy = csc_matrix(chunked_tensor_scipy)
    chunked_tensor_scipy_dict = {
        "data": chunked_tensor_scipy.data,
        "indices": chunked_tensor_scipy.indices,
        "indptr": chunked_tensor_scipy.indptr,
        "num_docs": num_chunks
    }

    print("Chunked tensor scipy: ", chunked_tensor_scipy, " Shape: ", chunked_tensor_scipy.shape)

    first_num_chunks_selected = 64
    target_retrieved = 32

    priority_target_compare = 16

    for query in queries:
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        baseline_retrieved_documents = retriever.retrieve(query_tokens, k=target_retrieved)
        baseline_retrieved_document_ids = baseline_retrieved_documents[0][0]

        query_tokens = bm25s.tokenization.convert_tokenized_to_string_list(query_tokens)[0]

        print("Query tokens: ", query_tokens)
        query_ids = retriever.get_tokens_ids(query_tokens)
        print("Query IDs: ", query_ids, " Length: ", len(query_ids))

        print("score data: ", chunked_tensor_scipy.data)

        top_chunks = _retrieve_numba_functional(
            query_tokens_ids=[query_ids],
            scores=chunked_tensor_scipy_dict,
            k=first_num_chunks_selected
        )
        print("Top chunks: ", top_chunks)

        selected_chunks = top_chunks[0][0]

        all_documents = []
        document_real_index_map = []

        for chunk_idx in selected_chunks:
            print("Chunk index: ", chunk_idx)
            relevant_documents = document_chunk_list[chunk_idx]
            index_range_start = chunk_idx * chunk_size
            index_range_end = min((chunk_idx + 1) * chunk_size, num_rows) # type: ignore

            document_real_index_map.extend(
                list(range(index_range_start, index_range_end))
            )
            all_documents.extend(relevant_documents)

        all_documents = torch.stack(all_documents, dim=0)

        print("All documents: ", all_documents, " Shape: ", all_documents.shape)
        all_documents = all_documents.to_sparse_csc()
        print("All documents sparse CSC: ", all_documents, " Shape: ", all_documents.shape)
        all_documents_scipy_dict = {
            "data": all_documents.values().cpu().numpy(),
            "indices": all_documents.row_indices().cpu().numpy(),
            "indptr": all_documents.ccol_indices().cpu().numpy(),
            "num_docs": len(document_real_index_map)
        }

        top_documents = _retrieve_numba_functional(
            query_tokens_ids=[query_ids],
            scores=all_documents_scipy_dict,
            k=target_retrieved
        )

        print("Top documents: ", top_documents)

        top_documents_indices = top_documents[0][0]
        actual_retrieved_document_indices = [
            document_real_index_map[idx] for idx in top_documents_indices
        ]

        print("Actual retrieved document indices: ", actual_retrieved_document_indices)
        print("Comparisons made: ", chunked_tensor_scipy_dict['num_docs'], " + ", all_documents_scipy_dict['num_docs'], " = ", chunked_tensor_scipy_dict['num_docs'] + all_documents_scipy_dict['num_docs'], " out of ", len(corpus_lst))
        print("Percentage comparisons: ", (chunked_tensor_scipy_dict['num_docs'] + all_documents_scipy_dict['num_docs']) / len(corpus_lst) * 100, "%")
        overlap = set(actual_retrieved_document_indices) & set(baseline_retrieved_document_ids.tolist())
        print("Overlap count of actual top k: ", len(overlap), " out of ", len(baseline_retrieved_document_ids))
        print("Overlap: ", overlap)

        print("Evaluating against the priority results: ", priority_target_compare)
        priority_retrieved_documents = set(actual_retrieved_document_indices[:priority_target_compare]) & set(baseline_retrieved_document_ids.tolist())
        print("Priority overlap count: ", len(priority_retrieved_documents), " out of ", priority_target_compare)
        

if __name__ == "__main__":
    main()