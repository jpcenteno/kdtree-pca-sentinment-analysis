#!/bin/bash

time for a in {10..90..10}; do for k in {100..1000..100}; do ./searcher.py --algorithm hill-climbing -k $k --alpha $a --k-step 20 --like-classify $1; done; done
