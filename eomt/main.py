# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import jsonargparse._typehints as _t
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
import sys
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"

# NOTA: save_weights_only=false è configurato nel YAML per salvare anche optimizer state
# Questo permette di riprendere il training con --resume_from senza KeyError

# Patch CheckpointConnector per gestire gracefully checkpoint senza optimizer state
# Questo evita KeyError quando si tenta di riprendere il training da checkpoint con solo pesi modello
from lightning.pytorch.trainer.connectors.checkpoint_connector import CheckpointConnector

_original_restore_optimizers = CheckpointConnector.restore_optimizers_and_schedulers

def _safe_restore_optimizers(self, checkpoint):
    """Sovrascrive restore_optimizers_and_schedulers per gestire checkpoint senza optimizer state."""
    if checkpoint is None:
        return _original_restore_optimizers(self, checkpoint)
    
    # Verifica se il checkpoint ha optimizer state
    has_optimizer = (
        "optimizer_states" in checkpoint 
        and checkpoint.get("optimizer_states") is not None
        and len(checkpoint.get("optimizer_states", [])) > 0
    )
    
    if not has_optimizer:
        # Checkpoint contiene solo pesi → salta ripristino optimizer (sarà re-inizializzato)
        logging.warning(
            "⚠️ Checkpoint contains only model weights (no optimizer state). "
            "Skipping optimizer restore - optimizer will be re-initialized and training will restart from epoch 0."
        )
        logging.info("💡 TIP: For proper resume, use checkpoints saved with save_weights_only=false")
        logging.info("💡 Or, extract weights using extract_model_weights.py and use as model.ckpt_path in YAML")
        
        # Reset epoch e global_step nel trainer per ripartire da capo
        if hasattr(self, "trainer") and self.trainer is not None:
            # Reset training state for fresh start
            if hasattr(self.trainer, "current_epoch"):
                self.trainer.current_epoch = 0
            if hasattr(self.trainer, "global_step"):
                self.trainer.global_step = 0
        
        # Salta ripristino optimizer (sarà re-inizializzato automaticamente)
        # Non sollevare KeyError, semplicemente ritorna None
        return None
    
    # Checkpoint completo → usa comportamento normale
    try:
        return _original_restore_optimizers(self, checkpoint)
    except KeyError as e:
        if "optimizer" in str(e).lower() or "optimizer_states" in str(e) or "only the model" in str(e).lower():
            # Errore relativo a optimizer state mancante → gestisci gracefully
            logging.warning(f"⚠️ Error restoring optimizer state: {e}")
            logging.info("💡 Optimizer will be re-initialized and training will restart from epoch 0")
            # Reset epoch e global_step
            if hasattr(self, "trainer") and self.trainer is not None:
                if hasattr(self.trainer, "current_epoch"):
                    self.trainer.current_epoch = 0
                if hasattr(self.trainer, "global_step"):
                    self.trainer.global_step = 0
            return None
        else:
            # Altro tipo di KeyError → rilancia
            raise

# Applica la patch globale al CheckpointConnector
CheckpointConnector.restore_optimizers_and_schedulers = _safe_restore_optimizers


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")
        # Use --resume_from instead of --ckpt_path to avoid conflict with Trainer.fit(ckpt_path=...)
        parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume training from (alternative to Trainer.ckpt_path)")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        # save_weights_only=false nel YAML assicura che i checkpoint includano optimizer state
        # Questo permette di riprendere il training con --resume_from senza KeyError
        
        # Get resume_from checkpoint path from command line argument if provided
        ckpt_path = None
        try:
            # Access --resume_from argument from parsed config
            if hasattr(self, "config") and isinstance(self.config, dict):
                subcommand = self.config.get("subcommand", "fit")
                subcommand_config = self.config.get(subcommand, {})
                if isinstance(subcommand_config, dict):
                    ckpt_path = subcommand_config.get("resume_from")
                if ckpt_path is None:
                    ckpt_path = self.config.get("resume_from")
            
            # Fallback: Parse from sys.argv directly
            if ckpt_path is None and "--resume_from" in sys.argv:
                idx = sys.argv.index("--resume_from")
                if idx + 1 < len(sys.argv):
                    ckpt_path = sys.argv[idx + 1]
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
            logging.debug(f"Could not access resume_from: {e}")
        
        if ckpt_path:
            logging.info(f"📂 Resuming training from checkpoint: {ckpt_path}")
            kwargs["ckpt_path"] = ckpt_path

        self.trainer.fit(model, **kwargs)


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "devices": 1,
            "gradient_clip_val": 0.3,  # Task 3A: Reduced from 1.0 - 0.3 is safer for AMP (0.3-0.5 range)
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
