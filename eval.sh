# deepspeed --include localhost:0 train_rio_sam_prune_new.py \
# --version="./LISA_finetune/RIO" \
# --dataset_dir='./dataset' \
# --vision_pretrained="./sam_vit_l_0b3195.pth" \
# --dataset="reason_seg" \
# --sample_rates="1" \
# --exp_name="lisa-7b-vit-l" \
# --val_dataset="RIO|val|uncommon" \
# --steps_per_epoch=1000 \
# --vit_type="vit-l" \
# --eval_only \
# --test_model_bin \
# --sam_component_dir "./LISA_finetune/LISA/ViT-L" \

deepspeed --include localhost:0 train_rio_sam_prune_new.py \
--version="./LISA_finetune/RIO" \
--dataset_dir='./dataset' \
--vision_pretrained="./sam_vit_b_01ec64.pth" \
--dataset="reason_seg" \
--sample_rates="1" \
--exp_name="lisa-7b-vit-b" \
--val_dataset="RIO|val|uncommon" \
--steps_per_epoch=1000 \
--vit_type="vit-b" \
--eval_only \
--test_model_bin \
--sam_component_dir "./LISA_finetune/LISA/ViT-B" \