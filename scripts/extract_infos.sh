#!/bin/bash
python extract_infos.py \
    --draw_img_save_dir='assets/result/det_dbpp/v2/raw' \
    --image_dir="assets/test/raw" \
    --det_model_dir="assets/inference/det_db" \
    --det_algorithm="DB++"