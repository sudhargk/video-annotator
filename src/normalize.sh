#!/bin/bash
function normalize(){
	head -n 1 $1 > $2
	tail -n +2 $1| shuf -n $2 >>$3
}
INPUT_FILE=$1
OUT_PATH=$2
TRIM=$3
tail -n +2 $INPUT_FILE > tmp.txt
BASE_PATH=$(dirname $INPUT_FILE)
mkdir -p $OUT_PATH
while read -r LINE; do
	if [ -n '$LINE' ]; then
		normalize $BASE_PATH'/'$LINE $OUT_PATH'/'$LINE $TRIM
	fi
done < tmp.txt


