import cv2
import onnxruntime as ort
import argparse
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=False, help="input image")
parser.add_argument('-b','--box', nargs='+', help='detection box', required=True)
args=parser.parse_args()

img_path = '/usr/src/tensorrt/data/ultraface/' + args.image
image = cv2.imread(img_path)
height, width, _ = image.shape
color = (255, 128, 0)
#print(args.box)
p1 = (int(float(args.box[0]) * width), int(float(args.box[1]) * height))
p2 = (int(float(args.box[2]) * width), int(float(args.box[3]) * height))
cv2.rectangle(image, p1, p2, color, 4)
cv2.imwrite("./detections/" + args.image, image)