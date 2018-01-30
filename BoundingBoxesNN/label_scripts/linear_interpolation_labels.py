"""
This script is going to linearly interpolate the labels created by
imglab stored in an xml file.
"""

import numpy as np

def get_attributes(string, c):
  # Finds attributes surrounded by c in string
  index = [pos for pos, char in enumerate(string) if char == c]
  attributes = []
  for i in range(len(index) // 2):
    attributes.append(string[(index[2 * i] + 1) : index[2 * i + 1]])
  return attributes

def read_image(file):
  # Read xml block describing image and extract attributes
  line = file.readline()
  image_name = None
  label = None
  label_coord = None
  if not line == '</images>\n':
    image_name = get_attributes(line, '\'')[0]
    line = file.readline()
    if not line == '  </image>\n':
      label_coord = [int(i) for i in get_attributes(line, '\'')]
      line = file.readline()
      label = get_attributes(line, '<')[0][6 :]
      line = file.readline()
      line = file.readline()
  return image_name, label, label_coord
  
def write_images(file, image_name, label, label_coord):
  # Write xml block describing image and attributes
  for i in range(len(image_name)):
    file.write('  <image file=\'%s\'>\n' % image_name[i])
    file.write('    <box top=\'%i\' left=\'%i\' width=\'%i\' height=\'%i\'>\n' % tuple(label_coord[i]))
    file.write('      <label>%s</label>\n' % label[i])
    file.write('    </box>\n  </image>\n')



# Set parameter
in_file_name = 'labels8.xml'
out_file_name = 'labels8_interpolated.xml'

# Read file
in_file = open(in_file_name, 'r')
out_file = open(out_file_name, 'w')

# Copy header
while True:
  line = in_file.readline()
  out_file.write(line)
  if line == '<images>\n':
    break

image_name, label, old_label_coord = read_image(in_file)
# Handle starting images without labels
while not old_label_coord:
  out_file.write('  <image file=\'%s\'>\n  </image>\n' % image_name)
  image_name, label, old_label_coord = read_image(in_file)
write_images(out_file, [image_name], [label], [old_label_coord])

# Process file
image_names = []
while True:
  image_name, label, label_coord = read_image(in_file)
  image_names.append(image_name)
  if image_name == None:
    break
  if not label == None:
    labels = len(image_names) * [label]
    ramp = np.linspace(0, 1, len(image_names), False)
    label_coords = np.outer(ramp, old_label_coord) + np.outer(1 - ramp, label_coord)
    label_coords = np.flip(label_coords, 0)
    label_coords = label_coords.astype(int).tolist()
    write_images(out_file, image_names, labels, label_coords)
    old_label_coord = label_coord
    image_names = []

# Copy footer
out_file.write('</images>\n')
out_file.write('</dataset>')

# Close files
in_file.close()
out_file.close()