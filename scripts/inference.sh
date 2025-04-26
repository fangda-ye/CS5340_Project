PYTHONPATH=. python src/inference_rnn.py \
    --rnn_weights_path ./output/rnn_v3_seq_model_output/rnn_v3_model_final.pth \
    --prompt "A chef passionately tossing pizza dough in a rustic kitchen, motion blur." \
    --output_dir ./inference_output/ \
    --base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --npnet_model_id SDXL \
    --text_embed_dim 1280 \
    --noise_resolution 128 \
    --noise_channels 4 \
    --cnn_base_filters 64 \
    --cnn_num_blocks 2 2 2 2 \
    --cnn_feat_dim 512 \
    --cnn_groups 8 \
    --gru_hidden_size 1024 \
    --gru_num_layers 2 \
    --predict_variance  \
    --num_gen_steps 10  \
    --num_inference_steps 30  \
    --guidance_scale 5.5  \
    --seed 42  \
    --generate_standard  \
    --dtype float16

PYTHONPATH=. python scripts/batch_inference.py \
    --prompt_file ./data/test_prompts.txt \
    --output_base_dir ./inference_output/ \
    --rnn_weights_path ./output/rnn_v3_seq_model_output/rnn_v3_model_final.pth \
    --base_model_id stabilityai/stable-diffusion-xl-base-1.0 \
    --start_seed 1000