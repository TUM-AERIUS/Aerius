import argparse
import re
import subprocess

from PIL import Image

REGEX_BOX = r".*<box.*>.*"
REGEX_T = r"top='(-?\d*)'"
REGEX_L = r"left='(-?\d*)'"
REGEX_W = r"width='(-?\d*)'"
REGEX_H = r"height='(-?\d*)'"
REGEX_IMG = r".*<image file=.*>.*"

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
                    f_out.write('    <box top=\'%i\' left=\'%i\' width=\'%i\' height=\'%i\'>\n' % (t,l,w,h))
                else:
                    f_out.write(line)
                    if re.match(REGEX_IMG, line):
                        filepath = (re.search(r"file='(.*\.\w*)'", line).group(1))
                        with Image.open(filepath) as img:
                            img_w, img_h = img.size
    if replace_input_file:
        subprocess.call(["mv", output_file, input_file])

# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser(description='Removes all invalid bounding box coordinates of a given imglab annotation file '
                                                 'by adjusting the bounding box coordinates accordingly')
    parser.add_argument('imglab_file', help='Path to imglab annotation file (.xml)')
    parser.add_argument('--output', '-o', help='Select an output file. By default the input file itself is modified.')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    remove_invalid_values(args.imglab_file, output_file=args.output)

if __name__ == '__main__':
    main()