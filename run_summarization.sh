python -u __main_completion__.py \
        --model_type alpha \
        --layers 3 \
        --e_ff_fold 4 \
        --attn_heads 8 \
        --hidden 1024 \
        --lan_embedding_dim 1024 \
        --projection_dim 2048 \
        --batch_size 64 \
        --accu_batch_size 128 \
        --val_batch_size 256 \
        --infer_batch_size 256 \
        --weight_decay 1e-5 \
        --lr 1e-4 \
        --min_lr 1e-5 \
        --patience 0 \
        --dropout 0.2 \
        --epochs 20 \
        --MultiStepLR False \
        --milestones 8 \
        --lr_scheduler True
