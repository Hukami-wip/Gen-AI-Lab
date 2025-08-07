# # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#


import asyncio
from pathlib import Path

import fire
from lcm.datasets.sentence_splitter_pipeline import (
    FullPipeline,
    FullPipelineConfig,
    SentenceSplitterConfig,
)
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import stopes_data_mapper
from stopes.modules.preprocess.sonar_text_embedding import (
    LangColumnConfig,
    SonarTextEmbedderConfig,
)
from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.hf_shards import HFInputConfig
from stopes.utils.sharding.parquet_shards import ParquetOutputConfig


def run(
    output_dir: str | Path,
    train_split: str = "train[0:10000]",
    val_split: str = "train[10000:11000]",
):
    """
    launch a preprocessing pipeline, this will use SAT to split text in sentences and then use SONAR to
    embed each sentence.
    This example downloads data from huggingface and outputs it to a parquet dataset.

    Args:
        output_dir: Directory where the processed data will be written
        train_split: HuggingFace split syntax for training data (e.g., "train[0:10000]")
        val_split: HuggingFace split syntax for validation data (e.g., "train[10000:11000]")
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # setup the sentence splitter with optimized batch sizes for GPU utilization
    splitter_config = SentenceSplitterConfig(
        columns=[
            "text"
        ],  # this is the column in the input dataset where we expect to find text to split
        model_name="sat-3l",
        verbose=True,
        sentence_threshold=0.2,  # sentence splitting threshold to tune based on the data (domain, language, etc.)
        max_sentence_len=256,
    )
    # setup SONAR, we are only going to deal with english
    sonar_encoder_config = SonarTextEmbedderConfig(
        column_config=[  # we can process several columns at once which is useful for finetuning datasets
            LangColumnConfig("text_sentences", lang_value="eng_Latn")
        ],  # splitter has output a new column `text_sentences` and this is what we will embed
        device="cuda",  # we want to work on a GPU, if you want to try this on a cpu, change the device here
        batch_size=256,  # Increase for better GPU utilization
    )
    # setup the full pipeline, that will use the splitter and the sonar embeddings,
    full_config = FullPipelineConfig(
        splitter_config=splitter_config,
        sonar_encoder_config=sonar_encoder_config,
    )

    print(f"Processing training data: {train_split}")
    print(f"Processing validation data: {val_split}")

    # setup the input to download from huggingface, adjust this to the dataset you care about
    # Checkout https://github.com/facebookresearch/stopes/tree/main/stopes/utils/sharding for other potential
    # input systems (jsonl, parquet) and how to configure them in this pipeline.

    # Process training data
    train_input_config = HFInputConfig(
        input_file="wikimedia/wikipedia",
        data_dir="20231101.en",
        split=train_split,
        num_shards=1,  # Single shard to avoid parallel processing
        batch_format=BatchFormat.ARROW,
        batch_size=5,  # adjust to your system's size
    )

    # setup the output to write to parquet
    train_output_config = ParquetOutputConfig(
        output_dir / "train",
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )

    # # Process validation data
    val_input_config = HFInputConfig(
        input_file="wikimedia/wikipedia",
        data_dir="20231101.en",
        split=val_split,
        num_shards=1,  # Single shard to avoid parallel processing
        batch_format=BatchFormat.ARROW,
        batch_size=5,
    )

    val_output_config = ParquetOutputConfig(
        output_dir / "validation",
        keep_same_partitioning=False,
        row_group_size=200,
        batch_size=200,
    )

    # requirements for our slurm jobs, if you are using a local cpu, you can ignore this
    # if you are using slurm but no gpus, remove the gpus_per_node config
    req = Requirements(
        mem_gb=120, gpus_per_node=1, cpus_per_task=10, timeout_min=3 * 24 * 60
    )
    # launching config, here we use `local` to run locally, but you can switch it to `slurm` if you have a SLURM cluster.
    launcher = Launcher(
        cache=None,
        cluster="local",
    )

    # Process training data first
    print("Processing training data...")
    stopes_wrapped = stopes_data_mapper(req, {"name": "prep_wiki_train"})(FullPipeline)
    train_module = stopes_wrapped(train_input_config, train_output_config, full_config)
    asyncio.run(launcher.schedule(train_module))
    print("Training data processing complete!")

    # Process validation data after training data is complete
    print("Processing validation data...")
    val_module = stopes_wrapped(val_input_config, val_output_config, full_config)
    asyncio.run(launcher.schedule(val_module))
    print("Validation data processing complete!")

    print(f"Data processing complete. Output saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(run)
