## This code is modified from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Script for downloading data
download_gg_large () {
  FID=$1
  DATAPATH=$2
  wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id="$FID -O tmp.html
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(cat tmp.html | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FID -O $DATAPATH
  rm -rf /tmp/cookies.txt tmp.html
}

DATAPATH=data

mkdir -p $DATAPATH
cd $DATAPATH
download_gg_large "1Jdkrq_gH0R5sGOYhXuhSLgQbjPCgDGGw" KVQA_annotations_val.json
download_gg_large "1vQXgKcIu6v9GBGIiKDa-L0FnimdiUAj_" KVQA_annotations_train.json
download_gg_large "1DmCALEiKyPV58kBXb4mxsUQBSaFxotbI" KVQA_annotations_test.json
download_gg_large "1OhDQSMxQcLrYEZAsV10Ik7wZrC835Wyj" check_kvqa.py

# below 2 files(word-embeddings.zip, ko.zip) needs to be downloaded manually
# download_gg_large "1gpOaOl0BcUvYpgoOA2JpZY2z-BUhuBLX" word-embeddings.zip
# ratsgo
# unzip word-embeddings.zip

# download_gg_large "0B0ZXk88koS2KbDhXdWg1Q2RydlU" ko.zip
wget "https://www.dropbox.com/s/stt4y0zcp2c0iyb/ko.tar.gz?dl=1" -O ko.tar.gz

# Kyubong Park
mkdir -p fasttext
tar -zxvf ko.tar.gz -C fasttext
# unzip ko.zip -d word2vec

# mkdir -p features
cd features
# Vizwiz
# https://drive.google.com/file/d/1sdCbVDnt8jVlGVhm89NbkWMjZE-INoKN/view?usp=sharing
download_gg_large "19lKxKb5pQtYuICNThAtxBsUWfAgr7Jdy" VizWiz_resnet101_faster_rcnn_genome.tsv
# KVQA
# https://drive.google.com/file/d/19lKxKb5pQtYuICNThAtxBsUWfAgr7Jdy/view?usp=sharing
download_gg_large "19lKxKb5pQtYuICNThAtxBsUWfAgr7Jdy" KVQA_resnet101_faster_rcnn_genome.tsv

cd ..
cd ..