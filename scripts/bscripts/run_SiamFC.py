import scripts.model.detector as detector
import time
import scipy.ndimage as ndimage
import json
import collections


def parse_arguments():
    with open('/home/ubuntu/src/siamfc-tf/parameters/hyperparams.json') as json_file:
        hp = json.load(json_file)
    with open('/home/ubuntu/src/siamfc-tf/parameters/evaluation.json') as json_file:
        ev = json.load(json_file)
    with open('/home/ubuntu/src/siamfc-tf/parameters/run.json') as json_file:
        run = json.load(json_file)
    with open('/home/ubuntu/src/siamfc-tf/parameters/environment.json') as json_file:
        env = json.load(json_file)
    with open('/home/ubuntu/src/siamfc-tf/parameters/design.json') as json_file:
        design = json.load(json_file)

    hp = collections.namedtuple('hp', hp.keys())(**hp)
    ev = collections.namedtuple('evaluation', ev.keys())(**ev)
    run = collections.namedtuple('run', run.keys())(**run)
    env = collections.namedtuple('env', env.keys())(**env)
    design = collections.namedtuple('design', design.keys())(**design)

    return hp, ev, run, env, design


hp, ev, run, env, design = parse_arguments()
siamfc_detector = detector.SiameseNetwork(hp, design, env)


def run_SiamFC(seq, results_path, save_image):
    global siamfc_detector
    seq_len = seq.endFrame - seq.startFrame + 1
    results = [[0., 0., 0., 0.]] * seq_len

    # Convert from matlab indices to py indices, and slice array from start to end frames, inclusive.
    start_frame = seq.startFrame - 1
    end_frame = seq.endFrame - 1
    seq_frames = seq.s_frames[start_frame : end_frame + 1]

    results[0] = seq.init_rect
    trkr = siamfc_detector
    trkr.set_target(_read_image(seq_frames[0]), seq.init_rect)

    start = time.time()
    for i, img_file in enumerate(seq_frames[1:], start=1):
        img = _read_image(img_file)
        results[i] = trkr.detect(img)
    ms = time.time() - start

    return {'res': results, 'type': 'rect', 'fps': seq_len / float(ms) }


def _read_image(fname):
    return ndimage.imread(fname)[:, :, ::-1]


