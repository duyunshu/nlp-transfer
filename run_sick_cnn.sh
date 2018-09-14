#!/bin/sh

# cnn

for i in 1 2 3 4 5
do
  # sent_1
  # train-w2v
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=True

  # google-w2v
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=True

  # rand
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=True

  # train-glove
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=True

  # stanford-glove
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_1 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=True


  # sent_2
  # train-w2v
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-w2v --trans_emb=True

  # google-w2v
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=google-w2v --trans_emb=True

  # rand
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=rand --trans_emb=True

  # train-glove
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=train-glove --trans_emb=True

  # stanford-glove
  # no trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=False
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=False
  # do trans emb
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=CONTRADICTION --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=True
  python3 ./train.py --train_sick=True --sent_col=sent_2 --transfer=True --sent_label=ENTAILMENT --num_iter=1 --batch_size=32 --emb_mode=stanford-glove --trans_emb=True
done
