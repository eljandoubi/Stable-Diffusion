import argparse

DATASETS = ("conceptual_captions", "imagenet", "celebahq", "hf_images")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the VAE and diffusion trainers."""
    parser.add_argument(
        "--experiment_name",
        help="Name of Experiment being Launched",
        required=True,
        type=str,
    )
    parser.add_argument("--wandb_run_name", required=True, type=str)
    parser.add_argument(
        "--working_directory",
        help="Working Directory where checkpoints and logs are stored, inside a folder labeled by the experiment name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--log_wandb",
        action=argparse.BooleanOptionalAction,
        help="Do you want to log to WandB?",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Pass name of checkpoint folder to resume training from",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--training_config",
        help="Path to config file for all training information",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model_config",
        help="Path to config file for all model information",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="What dataset do you want to train on? hf_images reads a Hugging Face image dataset named by --path_to_dataset",
        choices=DATASETS,
        required=True,
        type=str,
    )
    parser.add_argument(
        "--path_to_dataset",
        help="Root directory of dataset, or the Hugging Face name for hf_images",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--path_to_save_gens",
        help="Folder you want to store the testing generations througout training",
        required=True,
        type=str,
    )
