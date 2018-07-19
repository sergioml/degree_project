#!/bin/sh
#!./local/bin/kaggle
FILE=$(ls -t /home/sergioml/degree_project/results/submissions/ | head -1)
echo $FILE
time /home/sergioml/.local/bin/kaggle competitions submit -c santander-product-recommendation -f /home/sergioml/degree_project/results/submissions/$FILE -m "$1"
