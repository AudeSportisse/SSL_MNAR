module load conda/5.0.1-python3.6
source activate virt_pytorch_conda

python3 script_main.py --valid_num 400 --general_method 'MLE' --seed 1 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 --prob_estim 0.0901 0.0993 0.0875 0.0885 0.0907 0.0895 0.0908 0.0906 0.0877 0.0916

python3 script_main.py --valid_num 400 --general_method 'MLE' --seed 2 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 --prob_estim 0.0901 0.0966 0.0909 0.0883 0.0934 0.0903 0.0906 0.0920  0.0926 0.0896

python3 script_main.py --valid_num 400 --general_method 'MLE' --seed 3 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 --prob_estim 0.0945 0.0927 0.0905 0.0875 0.0881 0.0939 0.0929 0.0915 0.0912 0.0916

python3 script_main.py  --valid_num 400 --general_method 'MLE' --seed 4 --lmbd 1 --cutoff_meth "meca" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 --prob_estim 0.0894 0.0923 0.0909 0.0922 0.0907 0.0901 0.0894 0.0916 0.0916 0.0943
