#!/bin/bash
for ((i=100;i<102;i++))
do 
python sMNIST.py --net-type RNN --nhid 512 --lr 0.0001 --alpha 0.9 --rinit xavier --permute --T 1 --random-seed $i --epochs 70
python sMNIST.py --net-type nnRNN --nhid 512 --lr 0.0002 --alpha 0.99 --rinit cayley --permute --T 1 --random-seed $i --epochs 70 --lr_orth 2e-5 --alam 0.1 --Tdecay 0.0001
python sMNIST.py --net-type expRNN --nhid 512 --lr 0.0005 --alpha 0.99 --rinit cayley --permute --T 1 --random-seed $i --epochs 70 --lr_orth 5e-5
python sMNIST.py --net-type LSTM --nhid 512 --lr 0.0005 --alpha 0.99 --rinit xavier --permute --T 1 --random-seed $i --epochs 70
done