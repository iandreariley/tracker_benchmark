import os
import sys
import cv2
import json
import numpy as np
import logging
import argparse

RESULTS_DIR = 'results/OPE'
CMT = 'CMT'
SIAM = 'SiamFC'

def get_predicted_bounding_boxes(sequence, tracker):
    with open(os.path.join('results', 'OPE', tracker, sequence + '.json'), 'rb') as results_file:
        results = json.load(results_file)
        return results[0]['res']


def intify(tup):
    return tuple(map(int, tup))


def draw_bbox(image, bbox, color):
    top_left = intify((bbox[0], bbox[1]))
    bottom_right = intify((bbox[0] + bbox[2], bbox[1] + bbox[3]))
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # draw bounding box.
    cv2.line(image, top_left, top_right, color, 4)
    cv2.line(image, top_right, bottom_right, color, 4)
    cv2.line(image, bottom_right, bottom_left, color, 4)
    cv2.line(image, bottom_left, top_left, color, 4)

parser = argparse.ArgumentParser("Turns directory of images into video")
parser.add_argument('directory', help='path to directory of images')
parser.add_argument('--extension', help='image extension (e.g. "jpg")', default='jpg')
parser.add_argument('--codec', help='video codec to use for compression', default='avc1')

args = parser.parse_args()
image_files = filter(lambda filename: filename.endswith(args.extension), os.listdir(os.path.join(args.directory, 'img')))
image_files = map(lambda filename: os.path.join(args.directory, 'img', filename), sorted(image_files, key=lambda filename: int(filename.split('.')[0])))
path_parts = args.directory.split('/')
sequence_name = path_parts[-1] or path_parts[-2]

if not len(sequence_name):
    logging.error("Somehow, magically, there is no sequence name in {0}. Exiting.".format(args.directory))
    sys.exit(1)

if not len(image_files):
    logging.error("Directory {0} contains no images with extension {1}.".format(args.directory, args.extension))
    sys.exit(1)

if args.codec == 'avc1':
    video_file_extension = 'h264'
elif args.codec == 'mp4v':
    video_file_extension = 'mp4'
else:
    logging.error("Codec {0} not supported by this program.".format(args.codec))
    sys.exit(1)
video_filename = sequence_name + '.' + video_file_extension
fourcc = cv2.cv.CV_FOURCC(*args.codec)
ratio = cv2.imread(image_files[0]).shape[:-1][::-1]
video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, ratio)
siam_predictions = get_predicted_bounding_boxes(sequence_name, SIAM)
cmt_predictions = get_predicted_bounding_boxes(sequence_name, CMT)
ground_truth = np.genfromtxt(os.path.join(args.directory, 'groundtruth_rect.txt'), delimiter=',')
import pdb; pdb.set_trace()
for i, image_file in enumerate(image_files):
    frame = cv2.imread(image_file)
    draw_bbox(frame, ground_truth[i], (255, 0, 0))
    draw_bbox(frame, siam_predictions[i], (0, 255, 0))
    draw_bbox(frame, cmt_predictions[i], (0, 0, 255))
    video_writer.write(frame)
    cv2.imshow("title", frame)
    cv2.waitKey(1)

video_writer.release()
