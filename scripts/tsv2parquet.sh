export NO_ALBUMENTATIONS_UPDATE=1
img2dataset --url_list data/Train_GCC-training_with_head.tsv \
            --input_format "tsv"\
            --url_col "url" \
            --caption_col "caption" \
            --output_format parquet \
            --output_folder data/GCC \
            --processes_count 32 \
            --thread_count 128 \
            --image_size 256 \
            --resize_mode keep_ratio