#!/bin/bash
# This scripts extracts the .h264 videos into images and creates train and validation annotations file accordingly
# Arguments:
#   1. Last training directory: Last annotation/video belonging to train (i.e. where to split train/val)
#   2. Last validation directory: Last annotation/video belonging to validation (i.e. where to split val/test)

if ([ "$#" -ne 2 ] || [ "$1" -gt "$2" ])
then
    echo "InvalidArgumentException: Please provide two arguments to this script, which specify the last training/validation directory respectively (i.e. where to split train, val and test)!" 1>&2
    exit 1
fi

# Convert all h264 files to png directories and move annotation files into it
for i in {1..20}; do
    declare f=labels"$i".xml
    declare f_d=photos_"$i"

    echo extracting video "$i"...

    mkdir "$f_d"
    ffmpeg -loglevel error -i test$i.h264 "$f_d"/image-%03d.png

    # Rename all object labels to "trousers"
    readarray -t annotations <<<"$(imglab -l $f)"
    for i in "${annotations[@]}"
    do
       imglab --rename "$i" "trousers" "$f"
    done
    imglab --rename "" "trousers" "$f" # handle unlabeled boxes

    cp "$f" "$f_d"
done

# Change the file path of all imagelab files to foldername_filename (currently hard coded due to different formats)
echo adjusting file paths...
for i in {1..5}; do
    sed -i -e "s/photos_${i}\/\(image\-\d*\)/photos_${i}_\1/g" photos_"$i"/labels"$i".xml
done
for i in {6..15}; do
	sed -i -e "s/\(image\-\d*\)/photos_${i}_\1/g" photos_"$i"/labels"$i".xml
done
for i in {16..20}; do
	sed -i -e "s/photos_${i}\/\(image\-\d*\)/photos_${i}_\1/g" photos_"$i"/labels"$i".xml
done

# Merge all annotations
# Create folder and empty annotations files
echo merging annotations...
mkdir merged
imglab -c train.xml
imglab -c val.xml
imglab -c test.xml
# Move images to merged and combine annotations
for i in {1..20} ; do
    # Set target annotation file accordingly
    declare target=train.xml
    if (($i>$2))
    then
        target=test.xml
    elif (($i>$1))
    then
        target=val.xml
    fi
	declare d=photos_"$i"
	imglab --add "$d"/labels"$i".xml "$target";
	mv merged.xml "$target";
	for f in `ls $d | grep .*\.png`; do
		mv "$d"/"$f" merged/"$d"_"$f";
	done;
done

# Move annotations to merged
mv train.xml merged/
mv val.xml merged/
mv test.xml merged/

# Cleanup
echo cleaning up...
for i in {1..20}; do
    rm -r photos_$i
done
echo done!
