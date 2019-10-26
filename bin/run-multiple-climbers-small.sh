#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 20 --alpha-step 2 \
              --grid-k {100..1000..100} --grid-alpha {90..10..10} \
              --use-dense-override \
              --out-history small \
              -imp $1


