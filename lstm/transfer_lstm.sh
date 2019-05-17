#!/bin/bash

# Set this to the IP address of the AWS instance
IP="34.222.26.215"

# # Upload inputs to training
# scp -i ../../cs230.pem setup_dirs.sh  ubuntu@${IP}:~
# scp -i ../../cs230.pem data.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem main.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem model.py  ubuntu@${IP}:~/lstm
# scp -i ../../cs230.pem ../data/text/lexicon.txt  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/corpus_train.txt  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/corpus_dev.txt  ubuntu@${IP}:~/data/text
# scp -i ../../cs230.pem ../data/text/corpus_test.txt  ubuntu@${IP}:~/data/text

# Download output from training
scp -i ../../cs230.pem ubuntu@${IP}:~/models/lstm.pt ../models