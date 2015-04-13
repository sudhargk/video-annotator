#!/bin/bash

function p_wait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

function to_int(){
	echo "($1+0.5)/1" | bc
}

DATA_PATH=$1
OUT_PATH=$2
MAX_JOBS=4
SHAPE_W=80 
SHAPE_H=60
WINDOW=5
WRITE_GRAY=0
WRITE_BGSUB=0

echo "Configuration INPUT_PATH: $DATA_PATH, OUT_PATH: $OUT_PATH"
echo "SHAPE: ($SHAPE_W,$SHAPE_H), WINDOW: $WINDOW, WRITE_GRAY: $WRITE_GRAY, WRITE_BGSUB: $WRITE_BGSUB"
echo "Starting Event Localization"
for CLASS_PATH in $DATA_PATH/*; do
	if [[ -d $CLASS_PATH ]]; then
		CLASS=$(basename $CLASS_PATH)
		echo "Processing class :: $CLASS"
		for input in $CLASS_PATH/*.avi;	do
			(
			video_path="$OUT_PATH/$CLASS/output/$(basename -s .avi $input).avi"
			feats_path="$OUT_PATH/$CLASS/feats/$(basename -s .avi $input).feat"
			python -W ignore run_EventTracker.py $input --extract $feats_path --write $video_path \
					--rsz_shape $SHAPE_W $SHAPE_H --window $WINDOW --write_gray $WRITE_GRAY \
					--write_bgsub $WRITE_BGSUB)&
			p_wait $MAX_JOBS
		done
	fi
done	
wait
echo "Event Localization [Done]"

if [ -nz $WRITE_GRAY ]; then
	NUM_CHANNELS=1;
else
	NUM_CHANNELS=3;
fi

if [ -nz $WRITE_BGSUB ]; then
	NUM_CHANNELS=`echo "$NUM_CHANNELS+1"|bc`;
fi

NUM_FEATS=`echo "$NUM_CHANNELS*$SHAPE_W*$SHAPE_H*$WINDOW"|bc`;
echo "Merging data ... NUM_FEATS:$NUM_FEATS"

for CLASS_PATH in $OUT_PATH/*; do
	(if [[ -d $CLASS_PATH ]]; then
		echo "Merging class :: $CLASS"
		CLASS=$(basename $CLASS_PATH)
		echo "Removing previous data files"
		rm -f "$OUT_PATH/train_$CLASS.txt"
		rm -f "$OUT_PATH/val_$CLASS.txt"
		rm -f "$OUT_PATH/test_$CLASS.txt"
		echo "$NUM_FEATS">"$OUT_PATH/train_$CLASS.txt"
		echo "$NUM_FEATS">"$OUT_PATH/val_$CLASS.txt"
		echo "$NUM_FEATS">"$OUT_PATH/test_$CLASS.txt"
		for input in $CLASS_PATH/feats/*.feat;	do
			shuf $input -o $input
			n=$(wc -l $input| cut -d" " -f1)
			n60=`echo "($n*0.6 +0.5)/1"|bc`;
			n80=`echo "($n*0.8 +0.5)/1"|bc`;
			n20=`echo "($n*0.2 +0.5)/1"|bc`;
			head $input -n $n60 >> "$OUT_PATH/train_$CLASS.txt"
			head $input -n $n80|tail -n $n20 >> "$OUT_PATH/val_$CLASS.txt"
			tail $input -n $n20 >> "$OUT_PATH/test_$CLASS.txt"			
		done
		print "Removing $CLASS_PATH/feats/"
		rm -rf "$CLASS_PATH/feats/"
	fi)&
	p_wait $MAX_JOBS
done
wait
echo "Merging data ... [DONE]"

