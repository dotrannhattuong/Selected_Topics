export  CUDA_VISIBLE_DEVICES=7
# infer
PYTHONPATH=$(pwd) python basicsr/test.py -opt options/Test/test_single_x4.yml
# gen result
python gen.py -f ./results/MSW/visualization/Single -s ./results/MSW.csv