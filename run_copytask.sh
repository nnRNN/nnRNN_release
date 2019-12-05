for ((i=0;i<70;i++))
do 
    python3 copytask.py --net-type RNN --nhid 128 --lr 0.001 --alpha 0.9 --rinit xavier --random-seed $i
    python3 copytask.py --net-type LSTM --nhid 128 --lr 0.0005 --alpha 0.99 --rinit xavier --random-seed $i
    python3 copytask.py --net-type expRNN --nhid 128 --lr 0.001 --lr_orth 0.0001 --alpha 0.99 --rinit henaff --random-seed $i
    python3 copytask.py --net-type nnRNN --nhid 128 --lr 0.0005 --lr_orth 0.000001 --alpha 0.99 --Tdecay 0.000001 --rinit henaff --random-seed $i
done