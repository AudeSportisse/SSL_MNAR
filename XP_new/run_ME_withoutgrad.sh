module load conda/5.0.1-python3.6
source activate virt_pytorch_conda

python3 script_main.py  --valid_num 400 --general_method 'ME_withoutgrad' --seed 1 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py  --valid_num 400 --general_method 'ME_withoutgrad' --seed 2 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py --valid_num 400 --general_method 'ME_withoutgrad' --seed 3 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py --valid_num 400 --general_method 'ME_withoutgrad' --seed 4 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 
