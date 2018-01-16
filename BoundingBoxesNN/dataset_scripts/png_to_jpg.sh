# Convert all images in the current folder from .png to .jpg
# Also adjusts any imglab files accordingly that are provided as arguments
mogrify -format jpg *.png
rm *.png
for var in "$@"
do
    sed -i -e 's/png/jpg/g' "$var"
done
