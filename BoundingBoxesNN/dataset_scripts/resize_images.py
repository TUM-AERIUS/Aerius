import argparse
import os
import re
import subprocess
import math

from PIL import Image, ImageOps

# Resize images with same aspect ratio and add black padding if necessary
def resize_pil_image_with_padding(img, width, height):
    # Get old image sizes
    img_w, img_h = img.size
    # Calculate new image sizes (with same aspect ratios)
    width_scale = 1.0 * width / img_w
    height_scale = 1.0 * height / img_h
    min_scale = min(width_scale, height_scale)
    new_img_w, new_img_h = int(math.ceil(img_w * min_scale)), int(math.ceil(img_h * min_scale))
    # Save new scale ratios of sizes (for altering bounding boxes later)
    resize_factors = (1.0 * new_img_w / img_w, 1.0 * new_img_h / img_h)
    # Resize image and add black padding on the right and bottom accordingly
    img = img.resize((new_img_w, new_img_h), Image.ANTIALIAS)
    padding = (0, 0, width - new_img_w, height - new_img_h)
    return ImageOps.expand(img, border=padding, fill='black'), resize_factors

class Resizer:

    def __init__(self, output_size):
        self.output_size = output_size

    # Resize individual image files
    def _resize_image(self, image):
        width, height = self.output_size
        file_extension = os.path.splitext(image)[-1]
        temp_file = os.path.join(os.path.dirname(image), ".resize_image_writing_temp_file." + file_extension)
        with Image.open(image) as img:
            img, resize_factors = resize_pil_image_with_padding(img, width, height)
            self.resize_factors = resize_factors
            img.save(temp_file)
        subprocess.call(["mv", temp_file, image])

    # Resize all bounding box coordinates of the given imglab file, as well as all corresponding images
    def resize_file(self, imglab_file):
        temp_file = os.path.join(os.path.dirname(imglab_file), ".resize_annotation_writing_temp_file")
        with open(temp_file, 'w+') as f_out:
            with open(imglab_file, 'r') as f_in:
                for line in f_in:

                    # Resize bounding box coordinates
                    if re.match(r".*<box.*>.*", line):
                        # Read box coordinates
                        t = int(re.search(r"top='(-?\d*)'", line).group(1))
                        l = int(re.search(r"left='(-?\d*)'", line).group(1))
                        w = int(re.search(r"width='(-?\d*)'", line).group(1))
                        h = int(re.search(r"height='(-?\d*)'", line).group(1))

                        # Resize box coordinates
                        w_resize_factor, h_resize_factor = self.resize_factors
                        t = int(t * h_resize_factor)
                        l = int(l * w_resize_factor)
                        w = int(w * w_resize_factor)
                        h = int(h * h_resize_factor)

                        # Write box coordinates
                        f_out.write("\t\t" + r"<box top='" + str(t) + r"' left='" + str(l)
                                    + r"' width='" + str(w) + r"' height='" + str(h) + "'>\n")

                    else:
                        f_out.write(line)

                        # Resize images
                        if re.match(r".*<image file=.*>.*", line):
                            filepath = (re.search(r"file='(.*\.\w*)'", line).group(1))
                            self._resize_image(filepath)

        subprocess.call(["mv", temp_file, imglab_file])


# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser(description='Resize all bounding boxes in given imglab annotation files, as well '
                                                 'as the corresponding images itself. During resizing, image ratios'
                                                 ' are kept constant and the shorter sides are padded with black.')
    parser.add_argument('width', help="Width of resized images")
    parser.add_argument('height', help='Height of resized images')
    parser.add_argument('imglab_files', help='Path to imglab annotation files (.xml)', nargs='+')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    output_size = (int(args.width), int(args.height))
    resizer = Resizer(output_size)
    for imglab_file in args.imglab_files:
        print("Resizing " + imglab_file + "...")
        resizer.resize_file(imglab_file)

if __name__ == '__main__':
    main()