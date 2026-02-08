uv run scripts/prepare_CC.sh --path_to_data_root data/GCC \
                  --path_to_save data/GCC/hf_train_unencoded \
                  --hf_clip_model_name openai/clip-vit-large-patch14 \
                  --hf_cache_dir data/GCC/hf_cache \
                  --cpu_batch_size 512 \
                  --gpu_batch_size 256 \
                  --num_cpu_workers 16 \
                  --pre_encode_text \
                  --dtype bfloat16