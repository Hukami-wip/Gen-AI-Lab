import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def flatten_dataset_streaming(input_dir, output_dir, sentences_per_file):
    """
    Flattens a nested dataset, computing article-level average embeddings
    and mapping each sentence to its article's average embedding.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])

    output_file_idx = 0
    sentence_embedding_buffer = []
    article_embedding_buffer = []

    for filename in tqdm(input_files, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        pf = pq.ParquetFile(input_path)

        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=["text_sentences_sonar_emb"])
            for article_embeddings in table.to_pydict()["text_sentences_sonar_emb"]:
                if not article_embeddings:
                    continue

                # Compute the average embedding for the article
                article_avg_embedding = np.mean(
                    np.array(article_embeddings), axis=0
                ).tolist()

                # Add each sentence and its corresponding article average to the buffer
                for sentence_embedding in article_embeddings:
                    sentence_embedding_buffer.append(sentence_embedding)
                    article_embedding_buffer.append(article_avg_embedding)

                # Write files whenever we have enough sentences
                while len(sentence_embedding_buffer) >= sentences_per_file:
                    chunk_sentences = sentence_embedding_buffer[:sentences_per_file]
                    chunk_articles = article_embedding_buffer[:sentences_per_file]

                    sentence_embedding_buffer = sentence_embedding_buffer[
                        sentences_per_file:
                    ]
                    article_embedding_buffer = article_embedding_buffer[
                        sentences_per_file:
                    ]

                    output_path = os.path.join(
                        output_dir, f"part-{output_file_idx:05d}.parquet"
                    )

                    sentence_array = pa.array(chunk_sentences)
                    article_array = pa.array(chunk_articles)

                    new_table = pa.Table.from_arrays(
                        [sentence_array, article_array],
                        names=["sentence_embedding", "article_embedding"],
                    )
                    pq.write_table(new_table, output_path)
                    output_file_idx += 1

    # Write any remaining embeddings in the buffer
    if sentence_embedding_buffer:
        output_path = os.path.join(output_dir, f"part-{output_file_idx:05d}.parquet")
        sentence_array = pa.array(sentence_embedding_buffer)
        article_array = pa.array(article_embedding_buffer)
        new_table = pa.Table.from_arrays(
            [sentence_array, article_array],
            names=["sentence_embedding", "article_embedding"],
        )
        pq.write_table(new_table, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten nested SONAR sentence embeddings into chunked files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the original parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the flattened parquet files.",
    )
    parser.add_argument(
        "--sentences_per_file",
        type=int,
        default=100_000,
        help="The number of sentences to save in each output parquet file.",
    )

    args = parser.parse_args()
    flatten_dataset_streaming(args.input_dir, args.output_dir, args.sentences_per_file)
