#!/bin/bash

function p_wait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

DATA_PATH="../examples/videos/ucf/"
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
			p_wait 4
		done
	fi
	wait
done	
echo "Event Localization [Done]"
