# import os

# import torch
# import torch.distributed as dist
from lightning.pytorch.cli import LightningCLI


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.sequence_length", "model.init_args.output_dim"
        )

    # def before_instantiate_classes(self) -> None:
    #     # Initialize distributed for Muon optimizer
    #     if not dist.is_initialized():
    #         # Set up for single GPU training
    #         os.environ["MASTER_ADDR"] = "localhost"
    #         os.environ["MASTER_PORT"] = "12355"
    #         os.environ["WORLD_SIZE"] = "1"
    #         os.environ["RANK"] = "0"
    #         os.environ["LOCAL_RANK"] = "0"

    #         # Actually initialize the process group
    #         if torch.cuda.is_available():
    #             dist.init_process_group(backend="nccl", world_size=1, rank=0)
    #         else:
    #             dist.init_process_group(backend="gloo", world_size=1, rank=0)


if __name__ == "__main__":
    CustomCLI(seed_everything_default=42)
