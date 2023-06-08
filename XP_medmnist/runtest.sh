module load conda/5.0.1-python3.6
source activate virt_pytorch_conda
pip install medmnist

python3 script_main_2D_dermaMNIST.py --lab_num 30 35 76 40 350 2000 20 --general_method 'DePl' --seed 0 --lmbd 1 --cutoff_meth "ICLR" 



