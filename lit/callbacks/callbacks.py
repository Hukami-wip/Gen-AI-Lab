import gc
import time

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline


class MetricsCallback(Callback):
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        total_params = sum(p.numel() for p in pl_module.parameters())
        pl_module.logger.experiment.add_scalar(
            "model_total_parameters", float(total_params), 0
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_start_time = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.monotonic() - self.train_batch_start_time
        batch_size = batch[0].size(0)
        throughput = batch_size / batch_time
        pl_module.log("train_throughput_samples_per_sec", throughput)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.val_batch_start_time = time.monotonic()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        latency = time.monotonic() - self.val_batch_start_time
        pl_module.log("validation_latency_sec", latency)

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self.test_batch_start_time = time.monotonic()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        latency = time.monotonic() - self.test_batch_start_time
        pl_module.log("test_latency_sec", latency)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Log final test metrics
        if trainer.is_global_zero:
            final_metrics = {
                k: v.item() for k, v in trainer.logged_metrics.items() if "test" in k
            }
            pl_module.logger.log_hyperparams(pl_module.hparams, final_metrics)


class SampleGeneratorCallback(Callback):
    """
    Callback to generate and print text samples from the GAN at a fixed interval.
    This callback carefully swaps models between the CPU and GPU to manage memory
    while still leveraging the GPU for faster inference.
    """

    def __init__(self, generation_interval: int = 2000):
        super().__init__()
        self.generation_interval = generation_interval

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ):
        """Swaps STAG and vec2text models to generate samples on GPU."""
        if ((trainer.global_step / 3 * 2) - 1) % self.generation_interval != 0:
            return

        if trainer.global_rank == 0:
            print(
                f"\n--- Swapping models to generate samples at step {trainer.global_step / 3 * 2} ---"
            )

            # 1. Move STAG model to CPU to free GPU memory
            original_device = pl_module.device
            pl_module.eval()
            pl_module.cpu()
            torch.cuda.empty_cache()
            print("Moved STAG model to CPU.")

            vec2text = None
            try:
                # 2. Load vec2text model onto the GPU
                print("Initializing EmbeddingToTextModelPipeline on GPU...")
                vec2text = EmbeddingToTextModelPipeline(
                    decoder="text_sonar_basic_decoder",
                    tokenizer="text_sonar_basic_encoder",
                    device=original_device,
                )
                print("EmbeddingToTextModelPipeline initialized successfully on GPU.")

                with torch.no_grad():
                    # 3. Prepare data on CPU for the STAG model
                    z_print_cpu = torch.randn(8, pl_module.hparams.z_dim, device="cpu")

                    # 4. Generate embeddings on the CPU
                    generated_embeddings_cpu, _ = pl_module.generator(z_print_cpu)

                    # 5. Move generated embeddings to GPU for vec2text
                    generated_embeddings_gpu = generated_embeddings_cpu.to(
                        original_device
                    ).float()

                    # 6. Generate text on GPU
                    generated_texts = vec2text.predict(
                        list(generated_embeddings_gpu), target_lang="eng_Latn"
                    )

                    for i, text in enumerate(generated_texts):
                        print(f"  Sample {i + 1}: '{text}'")
                    print("--------------------------------------------------\n")

                    # 7. Clean up tensors
                    del z_print_cpu
                    del generated_embeddings_cpu, generated_embeddings_gpu

            except Exception as e:
                import traceback

                print(f"ERROR in SampleGeneratorCallback: {e}")
                traceback.print_exc()
            finally:
                # 8. Clean up vec2text model to free GPU memory
                if vec2text is not None:
                    del vec2text
                    print("Cleaned up EmbeddingToTextModelPipeline.")

                torch.cuda.empty_cache()

                # 9. Move STAG model back to GPU and set to train mode
                pl_module.to(original_device)
                pl_module.train()
                print("Moved STAG model back to GPU.")

                # 10. Force garbage collection to clean up host RAM
                gc.collect()
                print("Forced garbage collection on host.")
