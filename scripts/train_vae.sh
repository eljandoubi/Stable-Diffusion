uv run accelerate launch --config_file config.yaml vae_trainer.py \
    --experiment_name VAETrainer \
    --wandb_run_name vae_cc \
    --working_directory work_dir/vae_cc \
    --training_config configs/vae_train.yaml \
    --model_config configs/ldm.yaml \
    --dataset conceptual_captions \
    --path_to_dataset data/GCC/hf_train_encoded \
    --path_to_save_gens "gens/conceptual_captions"