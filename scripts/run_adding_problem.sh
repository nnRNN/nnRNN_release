#!/bin/bash
for ((i=840;i<846;i++))
do 
python3 ../adding_problem.py --random-seed $i --c-length 100 --no-of-ones 2 --nhid 128 --net-type RNN --nonlin tanh
python3 ../adding_problem.py --random-seed $i --c-length 100 --no-of-ones 2 --nhid 512 --net-type nnRNN
python3 ../adding_problem.py --random-seed $i --c-length 100 --no-of-ones 2 --nhid 128 --net-type LSTM
python3 ../adding_problem.py --random-seed $i --c-length 100 --no-of-ones 2 --nhid 512 --net-type expRNN
done