#env difbir
export  CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONPATH=$(pwd) python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/Train/train_DAT_x4.yml --launcher pytorch
