#!/bin/bash
for ((i=200;i<210;i++))
do 
python3 copytask.py --net-type nnRNN --nhid 128 --lr 0.0005 --lr_orth 0.000001 --alpha 0.99 --Tdecay 0.000001 --rinit random --random-seed $i > output_logs/output_$i.txt
done