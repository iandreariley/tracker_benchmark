def run_CMT(seq, result_path, save_image):
    seq_len = seq.endFrame - seq.startFrame + 1
    return {'res': [[0, 0, 5, 5]] * seq_len, 'type': 'rect', 'fps': 1}