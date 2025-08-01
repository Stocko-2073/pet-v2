#!/bin/bash
echo "Rendering $1"
mkdir -p gif
cd gif
openscad -q --animate_sharding 1/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 2/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 3/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 4/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 5/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 6/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 7/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 8/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 9/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
openscad -q --animate_sharding 10/10 --imgsize 1920,1920 --colorscheme Starnight --animate 288 --camera $2 ../$1 &
wait
FILENAME=$(basename -- "$1")
DEST_MKV=../devlog/$FILENAME.`date +%Y%m%d%H%M%S`.mkv
DEST_MP4=../devlog/$FILENAME.`date +%Y%m%d%H%M%S`.mp4
ffmpeg -framerate 60 -pattern_type glob -i "*.png" $DEST_MKV
ffmpeg -i "$DEST_MKV" \
  -vf "yadif,format=yuv420p" \
  -r 60 \
  -c:v libx264 -preset fast -profile:v main -level 4.0 -crf 22 \
  -x264-params "keyint=100:min-keyint=10:ref=2:8x8dct=0:weightp=1:subme=6:vbv-bufsize=25000:vbv-maxrate=20000:rc-lookahead=30" \
  "$DEST_MP4"
rm $DEST_MKV
cd ..
rm -rf gif
prowl "MP4'd" "MP4'd"
