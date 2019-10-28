#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 2 --alpha-step 2 \
              --grid-k {1600..1650..5} --grid-alpha {450..550..10} \
              --use-dense-override \
              --out-history smaller1 \
              -imp $1
