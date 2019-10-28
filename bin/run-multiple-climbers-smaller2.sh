#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 1 --alpha-step 1 \
              --grid-k {1650..1725..3} --grid-alpha {250..325..5} \
              --use-dense-override \
              --out-history smaller2 \
              -imp $1
