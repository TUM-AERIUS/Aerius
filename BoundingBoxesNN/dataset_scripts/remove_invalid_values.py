import argparse
import re
import subprocess
import sys

from PIL import Image

REGEX_IMG = r".*<image file='(.*\.\w*)'>.*"
REGEX_BOX = r".*<box.*>.*"
REGEX_T = r"top='(-?\d*)'"
REGEX_L = r"left='(-?\d*)'"
REGEX_W = r"width='(-?\d*)'"
REGEX_H = r"height='(-?\d*)'"

# Remove all invalid values from the given input_file and save result as output_file
def remove_invalid_values(input_file, output_file=None):
    replace_input_file = (output_file is None or output_file == input_file)
    output_file = input_file + ".r_i_v_input_copy" if replace_input_file else output_file
    img_w, img_h = None, None
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w+') as f_out:
            for line in f_in:
                if img_w and img_h and re.match(REGEX_BOX, line):
                    t = int(re.search(REGEX_T, line).group(1))
                    l = int(re.search(REGEX_L, line).group(1))
                    w = int(re.search(REGEX_W, line).group(1))
                    h = int(re.search(REGEX_H, line).group(1))
                    notneg = lambda x : max(x, 0)
                    t, l, w, h = notneg(t), notneg(l), notneg(w), notneg(h)
                    w = min(w, img_w - l)
                    h = min(h, img_h - t)
                    f_out.write('\t\t<box top=\'%i\' left=\'%i\' width=\'%i\' height=\'%i\'>\n' % (t,l,w,h))
                else:
                    f_out.write(line)
                    if re.match(REGEX_IMG, line):
                        filepath = (re.search(REGEX_IMG, line).group(1))
                        with Image.open(filepath) as img:
                            img_w, img_h = img.size
    if replace_input_file:
        subprocess.call(["mv", output_file, input_file])

# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser(description='Removes all invalid bounding box coordinates of given imglab '
                                                 'annotation files by adjusting bounding box coordinates accordingly.')
    parser.add_argument('imglab_files', nargs='+', help='Path(s) to imglab annotation file(s) (.xml)')
    parser.add_argument('--outputs', '-o', nargs='*',
                        help='Select output files. By default, the input files are modified directly.')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    imglab_files = args.imglab_files
    output_files = args.outputs

    # Exit if -o or -l was given but number of output paths does not match number of inputs
    if output_files is not None and len(output_files) != len(imglab_files):
        sys.exit("InvalidArgumentException: -o flag was given, but number of outputs does not match number of inputs. "
                 "Please provide one output path for each input file.")

    output_files = [None]*len(imglab_files) if output_files is None else output_files
    for i, imglab_file in enumerate(imglab_files):
        print("Processing " + imglab_file + "...")
        remove_invalid_values(imglab_file, output_file=output_files[i])


if __name__ == '__main__':
    main()
