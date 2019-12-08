#!/bin/bash
python sMNIST.py --net-type RNN --nhid 512 --lr 0.0001 --alpha 0.9 --rinit xavier --permute --T 1 --random-seed 20 --epochs 70