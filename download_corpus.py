from bm25s.utils.beir import download_dataset, load_jsonl, load_corpus, load_queries
import typer
import json
from pathlib import Path


def main(
    dataset: str = typer.Argument(..., help="BEIR dataset name (e.g., scifact, nfcorpus, etc.)"),
    output_dir: str = typer.Option("data", help="Output directory for the converted JSON file"),
    datasets_dir: str = typer.Option("./datasets", help="Directory to store downloaded BEIR datasets"),
    max_queries: int = typer.Option(None, help="Maximum number of queries to include (for testing)"),
    max_documents: int = typer.Option(None, help="Maximum number of documents to include (for testing)"),
    redownload: bool = typer.Option(False, help="Force redownload of dataset"),
    show_progress: bool = typer.Option(True, help="Show progress bars"),
):
    """
    Download a BEIR dataset and convert it to the JSON format expected by the Rust main.rs.

    The output will be a JSON file with the structure:
    {
        "documents": ["doc1 text", "doc2 text", ...],
        "queries": ["query1 text", "query2 text", ...]
    }
    """
    print(f"📥 Downloading BEIR dataset: {dataset}")

    # Download the dataset
    try:
        dataset_path = download_dataset(
            dataset,
            save_dir=datasets_dir,
            redownload=redownload,
            show_progress=show_progress
        )
        print(f"✅ Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise typer.Exit(1)

    # Load corpus (documents)
    print(f"📖 Loading corpus...")
    try:
        corpus_dict = load_corpus(
            dataset,
            save_dir=datasets_dir,
            show_progress=show_progress,
            return_dict=True
        )
        print(f"✅ Loaded {len(corpus_dict)} documents")
    except Exception as e:
        print(f"❌ Error loading corpus: {e}")
        raise typer.Exit(1)

    # Load queries
    print(f"🔍 Loading queries...")
    try:
        queries_dict = load_queries(
            dataset,
            save_dir=datasets_dir,
            show_progress=show_progress,
            return_dict=True
        )
        print(f"✅ Loaded {len(queries_dict)} queries")
    except Exception as e:
        print(f"❌ Error loading queries: {e}")
        raise typer.Exit(1)

    # Convert corpus to list of strings (combine title and text)
    documents = []
    for doc_id, doc in corpus_dict.items():
        # Combine title and text as is common practice
        title = doc.get('title', '').strip()
        text = doc.get('text', '').strip()

        if title and text:
            combined_text = f"{title} {text}"
        elif title:
            combined_text = title
        elif text:
            combined_text = text
        else:
            combined_text = ""  # Empty document, but we'll keep it

        documents.append(combined_text)

        # Limit documents if specified
        if max_documents and len(documents) >= max_documents:
            break

    # Convert queries to list of strings
    queries = []
    for query_id, query in queries_dict.items():
        query_text = query.get('text', '').strip()
        if query_text:
            queries.append(query_text)

        # Limit queries if specified
        if max_queries and len(queries) >= max_queries:
            break

    # Create the output data structure
    output_data = {
        "documents": documents,
        "queries": queries
    }

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    output_file = output_path / f"{dataset}.json"
    print(f"💾 Saving converted dataset to: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved {len(documents)} documents and {len(queries)} queries")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")
        raise typer.Exit(1)

    print(f"🎉 Conversion complete! You can now use: cargo run -- --dataset-source {dataset}")


if __name__ == "__main__":
    typer.run(main)
