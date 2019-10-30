#!/bin/bash

tar --exclude="*pca_db*" --exclude=".git"  --exclude="doc" \
	--exclude="*imdb*" --exclude="*__pycache__*" \
	--exclude="*.clangd*" \
	--exclude="*.so" --exclude="*.o" --exclude="*.ipynb_check*" \
	--exclude="TODO.org" \
	-zcf metnum.rtp2.tar.gz *
