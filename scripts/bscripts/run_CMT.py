import numpy as np
import scripts.model.detector as detector
import scipy.ndimage as ndimage
import time


def run_CMT(seq, result_path, save_image):
    seq_len = seq.endFrame - seq.startFrame + 1
    results = [[0., 0., 0., 0.]] * seq_len

    # Convert from matlab indices to py indices, and slice array from start to end frames, inclusive.
    start_frame = seq.startFrame - 1
    end_frame = seq.endFrame - 1
    seq_frames = seq.s_frames[start_frame : end_frame + 1]

    results[0] = seq.init_rect

    trkr = detector.CmtDetector()
    trkr.set_target(_read_image(seq_frames[0]), seq.init_rect)

    start = time.time()
    for i, img_file in enumerate(seq_frames[1:], start=1):
        img = _read_image(img_file)
        results[i] = trkr.detect(img)
    ms = time.time() - start

    return {'res': results, 'type': 'rect', 'fps': seq_len / float(ms) }


def _read_image(fname):
    return ndimage.imread(fname, mode='RGB')[:, :, ::-1]
