export  CUDA_VISIBLE_DEVICES=7
# infer
PYTHONPATH=$(pwd) python basicsr/test.py -opt options/Test/test_single_x4.yml
# gen result
python gen.py -f ./results/test_single_x4_100k/visualization/Single -s ./results/DAT_scale_100k.csv