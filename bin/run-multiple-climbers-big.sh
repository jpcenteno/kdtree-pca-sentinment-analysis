#!/bin/bash

time for a in {500..1100..100}; do for k in {100..1100..100}; do ./searcher.py --algorithm hill-climbing -k $k --alpha $a --k-step 10 --alpha-step 10 --like-classify sklearn; done; done
