#!/bin/bash

function p_wait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

function to_int(){
	echo "($1+0.5)/1" | bc
}

DATA_PATH="../examples/videos/ucf/51"
OUT_PATH="test_results"
MAX_JOBS=4

echo "Starting Event Localization"
for CLASS_PATH in $DATA_PATH/*; do
	if [[ -d $CLASS_PATH ]]; then
		CLASS=$(basename $CLASS_PATH)
		echo "Processing class :: $CLASS"
		for input in $CLASS_PATH/*.avi;	do
			(video_path="$OUT_PATH/$CLASS/output/$(basename -s .avi $input).avi"
			feats_path="$OUT_PATH/$CLASS/feats/$(basename -s .avi $input).feat"
			python -W ignore run_EventLocalization.py $input --extract $feats_path --write $video_path)&
			p_wait $MAX_JOBS
		done
	fi
done	
wait
echo "Event Localization [Done]"

NUM_FEATS=54000
echo "Merging data ..."
for CLASS_PATH in $OUT_PATH/*; do
	if [[ -d $CLASS_PATH ]]; then
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
	fi
done

echo "Merging data ... [DONE]"
