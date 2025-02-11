deepspeed --include localhost:0  train_ds.py --version="xinlai/LISA-7B-v1-explanatory" 
--dataset_dir='./dataset'  
--vision_pretrained="./sam_vit_h_4b8939.pth"  
--dataset="reason_seg" 
--sample_rates="1" 
--exp_name="lisa-7b"
--epochs= 1 
--steps_per_epoch= 1000

deepspeed --include localhost:0  train_rio_ds.py --version="xinlai/LISA-7B-v1-explanatory" 
--dataset_dir='./dataset'  
--vision_pretrained="./sam_vit_h_4b8939.pth"  
--dataset="reason_seg" 
--sample_rates="1" 
--exp_name="lisa-7b"
--epochs= 1
--steps_per_epoch= 1000