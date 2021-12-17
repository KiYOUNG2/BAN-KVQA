#!/bin/bash

# Download Data
mkdir data
cd data
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MRYiyz54XKbcyQXFeCQe3f5VITrHRjSt' -O dictionary_kkma.kvqa.pkl

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1w5ffc4FER6gkOWZs6rOZpHWS88ttwnpf' -O ft_init.kvqa.npy

mkdir fasttext
cd fasttext
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=15fwTru453_Z89zOpoRMmyFLnDRYhxuzk' -O ko.vec
cd ..

mkdir cache
cd cache
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CgyHMi_T4hCqS1jqECTaGO7zvQzCvrDP' -O trainval_label2ans.kvqa.pkl
cd ..
cd ..

# Download Model weights
mkdir saved_models
cd saved_models
mkdir ban-kvqa-fasttext-pkb
mkdir ban-kvqa-roberta-base-rnn

cd ban-kvqa-fasttext-pkb
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r8kpK3gnfDrsHU8QZVCgOhhqnyh9BSDg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r8kpK3gnfDrsHU8QZVCgOhhqnyh9BSDg" -O ban-kvqa-fasttext-pkb.pth && rm -rf /tmp/cookies.txt
cd ..

cd ban-kvqa-roberta-base-rnn
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FApS-TWSXdaRRPKOpfC8IUL4D7QM0yrT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FApS-TWSXdaRRPKOpfC8IUL4D7QM0yrT" -O ban-kvqa-roberta-base-rnn.pth && rm -rf /tmp/cookies.txt
