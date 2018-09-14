#!/bin/sh

# cnn

for i in 1 2 3 4 5
do

  # freeze
  python3 ./train.py --train_sick=False --transfer=False --num_iter=1 --batch_size=32 --emb_mode=train-w2v --finetune=False --multi_label=False
  python3 ./train.py --train_sick=False --transfer=False --num_iter=1 --batch_size=32 --emb_mode=google-w2v --finetune=False --multi_label=False
  python3 ./train.py --train_sick=False --transfer=False --num_iter=1 --batch_size=32 --emb_mode=train-glove --finetune=False --multi_label=False
  python3 ./train.py --train_sick=False --transfer=False --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --finetune=False --multi_label=False
  python3 ./train.py --train_sick=False --transfer=False --num_iter=1 --batch_size=32 --emb_mode=rand --finetune=False --multi_label=False

  # finetune
  python3 ./train.py --train_sick=False --transfer=True --num_iter=1 --batch_size=32 --emb_mode=train-w2v --finetune=True --multi_label=True --trans_emb=False
  python3 ./train.py --train_sick=False --transfer=True --num_iter=1 --batch_size=32 --emb_mode=google-w2v --finetune=True --multi_label=True --trans_emb=False
  python3 ./train.py --train_sick=False --transfer=True --num_iter=1 --batch_size=32 --emb_mode=train-glove --finetune=True --multi_label=True --trans_emb=False
  python3 ./train.py --train_sick=False --transfer=True --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --finetune=True --multi_label=True --trans_emb=False
  python3 ./train.py --train_sick=False --transfer=True --num_iter=1 --batch_size=32 --emb_mode=rand --finetune=true --multi_label=True --trans_emb=False

done
