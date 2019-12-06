#!/bin/bash
for ((i=40;i<60;i++))
do 
python3 copytask.py --net-type nnRNN --gamma-zero-gradient True --nhid 128 --lr 0.0005 --lr_orth 0.000001 --alpha 0.99 --Tdecay 0.000001 --rinit henaff --random-seed $i
done
