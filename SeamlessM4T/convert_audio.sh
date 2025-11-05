#!/bin/bash

MELD_DIR=

for split in test; do
	for file in ${MELD_DIR}/video/${split}/*.mp4; do
		echo "Processing $file..."
		base=$(basename "$file" .mp4)
		ffmpeg -i "$file" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${MELD_DIR}/audio/${split}/${base}.wav"
	done
done

