import argparse
import json
import sys

import torch
import yaml
from PIL import Image
from safetensors.torch import save_file

import ldm_sample
import ldm_trainer
from modules import VAE, LDMConfig

MODEL_CONFIG = {
    "vae": {
        "img_size": 32,
        "latent_channels": 4,
        "vae_channels_per_block": [16, 32],
    },
    "unet": {
        "down_block_types": ["AttnDown", "Down"],
        "mid_block_types": "AttnMid",
        "up_block_types": ["Up", "AttnUp"],
        "unet_channels_per_block": [32, 64],
        "time_embed_start_dim": 16,
        "time_embed_proj_dim": 64,
        "groupnorm_groups": 4,
        "attention_head_dim": 8,
    },
    "diffusion": {"num_diffusion_timesteps": 20},
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "total_training_iterations": 2,
    "checkpoint_iterations": 2,
    "per_gpu_batch_size": 2,
    "gradient_accumulations_steps": 1,
    "max_grad_norm": 1.0,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.999,
    "optimizer_weight_decay": 0.0,
    "ema_decay": 0.9,
    "scale_factor_batches": 2,
    "sample_freq": 2,
    "num_samples": 2,
    "sample_steps": 2,
    "guidance_scale": 1.0,
    "pin_memory": False,
    "num_workers": 0,
    "random_resize": False,
    "interpolation": "bilinear",
    "random_flip_p": 0.5,
}


def test_latent_resolution_and_yaml_sections(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(MODEL_CONFIG))
    config = LDMConfig.from_yaml(str(path))
    assert config.latent_resolution == 16
    assert config.groupnorm_groups == 4
    assert config.num_diffusion_timesteps == 20


def test_train_then_sample_end_to_end_on_cpu(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(4):
        Image.new("RGB", (40, 40), color=(30 * i, 60, 90)).save(data / f"{i}.png")

    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump(MODEL_CONFIG))
    config = LDMConfig.from_yaml(str(model_yaml))

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    save_file(VAE(config).state_dict(), str(vae_dir / "model.safetensors"))

    training_yaml = tmp_path / "training.yaml"
    training_yaml.write_text(
        yaml.safe_dump(
            {"training_args": {**TRAINING_CONFIG, "vae_checkpoint": str(vae_dir)}}
        )
    )

    args = argparse.Namespace(
        experiment_name="ldm",
        wandb_run_name="ldm",
        working_directory=str(tmp_path / "work"),
        log_wandb=False,
        resume_from_checkpoint=None,
        training_config=str(training_yaml),
        model_config=str(model_yaml),
        dataset="celebahq",
        path_to_dataset=str(data),
        path_to_save_gens=str(tmp_path / "gens"),
    )
    ldm_trainer.train(args)

    experiment = tmp_path / "work" / "ldm"
    assert (
        json.loads((experiment / "scale_factor.json").read_text())["scale_factor"] > 0
    )
    assert (experiment / "checkpoint_2" / "ema.safetensors").exists()
    assert (experiment / "checkpoint_2" / "model.safetensors").exists()
    assert (tmp_path / "gens" / "iteration_2.png").exists()

    result_file = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ldm_sample.py",
            "--model_config",
            str(model_yaml),
            "--vae_checkpoint",
            str(vae_dir),
            "--ldm_checkpoint",
            str(experiment / "checkpoint_2"),
            "--num_images",
            "2",
            "--num_steps",
            "2",
            "--batch_size",
            "2",
            "--grid_out",
            str(tmp_path / "grid.png"),
            "--result_file",
            str(result_file),
        ],
    )
    ldm_sample.main()
    assert (tmp_path / "grid.png").exists()
    assert json.loads(result_file.read_text())["num_images"] == 2


def test_condition_dropout_uses_the_null_label_and_empty_context():
    config = LDMConfig(class_conditioning=True, num_classes=5, cfg_dropout_prob=1.0)
    labels, context, mask = ldm_trainer.drop_conditions(
        {"class_conditioning": torch.tensor([0, 3])}, config
    )
    assert labels.tolist() == [5, 5]
    assert context is None and mask is None

    config = LDMConfig(text_conditioning=True, cfg_dropout_prob=1.0)
    batch = {
        "text_conditioning": torch.randn(2, 4, 8),
        "text_attention_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    labels, context, mask = ldm_trainer.drop_conditions(batch, config)
    assert labels is None
    assert torch.count_nonzero(context) == 0
    assert mask.sum(dim=1).tolist() == [1, 1]
    assert batch["text_conditioning"].abs().sum() > 0

    unchanged = ldm_trainer.drop_conditions({"images": torch.zeros(1)}, config)
    assert unchanged == (None, None, None)


def test_ema_tracks_the_model_weights():
    model = torch.nn.Linear(2, 2)
    ema = ldm_trainer.EMA(model, decay=0.5)
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], before + 0.5)
    copy = torch.nn.Linear(2, 2)
    ema.copy_to(copy)
    assert torch.allclose(copy.weight, ema.shadow["weight"])
