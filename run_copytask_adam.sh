#!/bin/bash
for ((i=20;i<40;i++))
do 
    python3 copytask.py --net-type nnRNN --nhid 128 --lr 0.0005 --lr_orth 0.000001 --optimizer Adam --Tdecay 0.000001 --rinit cayley --random-seed $i
done

