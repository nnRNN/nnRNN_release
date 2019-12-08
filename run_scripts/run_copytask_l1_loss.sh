#!/bin/bash
for ((i=300;i<318;i++))
do
python3 copytask.py --net-type nnRNN --nhid 256 --lr 0.0005 --lr_orth 0.000001 --alpha 0.99 --Tdecay 0.000001 --rinit henaff --random-seed $i
done
