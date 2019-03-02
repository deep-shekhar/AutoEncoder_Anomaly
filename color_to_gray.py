import os
import sys
import cv2

directory = sys.argv[1]

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  img = cv2.imread(os.path.join(directory, file_name))
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(os.path.join(directory, file_name),img_gray)

print("All done")
