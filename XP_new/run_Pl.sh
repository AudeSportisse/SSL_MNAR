module load conda/5.0.1-python3.6
source activate virt_pytorch_conda

#python3 script_cifar10_2.py --mecha 'MNAR' --valid_num 400 --general_method 'MCAR_SSL_biased' --seed 1 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 400 517 0 862 1113 1438 1857 2398 3097 4000 --lab_num 400 310 30 186 144 111 86 67 52 40 
###seed 1,2,3,4
#python3 script_cifar10_2.py --mecha 'MNAR' --valid_num 400 --general_method 'MCAR_SSL_biased' --seed 1 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 10 517 667 862 1113 1438 1857 2398 3097 4000 --lab_num 40 310 240 186 144 111 86 67 52 40 
###seed 1,2,3,4

python3 script_main.py  --valid_num 400 --general_method 'Pl' --seed 1 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py  --valid_num 400 --general_method 'Pl' --seed 2 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py  --valid_num 400 --general_method 'Pl' --seed 3 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

python3 script_main.py --valid_num 400 --general_method 'Pl' --seed 4 --lmbd 1 --cutoff_meth "ICLR" --unlab_num 1635 1635 1635 1635 1635 1635 1635 1635 1635 1635  --lab_num 164 164 164 164 164 164 164 164 164 164 

