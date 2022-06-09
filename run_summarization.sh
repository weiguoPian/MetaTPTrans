CUDA_VISIBLE_DEVICES=0,1 nohup python -u __main_summarization__.py \
        --model_type beta \
        --layers 3 \
        --e_ff_fold 4 \
        --attn_heads 8 \
        --hidden 1024 \
        --lan_embedding_dim 1024 \
        --projection_dim 2048 \
        --batch_size 32 \
        --accu_batch_size 128 \
        --val_batch_size 256 \
        --infer_batch_size 256 \
        --lr 1e-4 \
        --min_lr 1e-5 > nohup.log 2>&1 &
