import cv2
import numpy as np
from deep_sort.multi_tracker import MultiTracker
from deep_sort import detection, nn_matching
from application_util import preprocessing, visualization
from deep_sort_app import gather_sequence_info, create_detections


def gather_multi_sequence_info(sequence_dirs, detection_files):
    seqs_info = [
        gather_sequence_info(seq_dir, det_file) for seq_dir, det_file in zip(sequence_dirs, detection_files)]
    for cam_idx, seq_info in enumerate(seqs_info):
        seq_info.update({"cam_idx": cam_idx})
    return seqs_info


def dummy_run(sequence_dirs, detection_files, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget):
    n_cams = len(sequence_dirs)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    multi_tracker = MultiTracker(metric, n_cams=n_cams)

    frame_idx = 0
    seqs_info = gather_multi_sequence_info(sequence_dirs, detection_files)
    while frame_idx < 10:
        detections = []
        for seq_info in seqs_info:
            dets_seq =  create_detections(seq_info["detections"], frame_idx, min_detection_height)
            dets_seq = [d for d in dets_seq if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in dets_seq])
            scores = np.array([d.confidence for d in dets_seq])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            dets_seq = [dets_seq[i] for i in indices]
            detections.append(dets_seq)

        # Update tracker.
        multi_tracker.predict()
        multi_tracker.update(detections)
        frame_idx += 1


if __name__ == '__main__':
    n_cams = 2
    sequence_dirs = n_cams * ['./MOT16/test/MOT16-06']
    detection_files = n_cams * ['./resources/detections/MOT16_POI_test/MOT16-06.npy']
    output_file = "/tmp/multi_hypotheses.txt"
    min_confidence = 0.3
    nms_max_overlap = 1.0,
    min_detection_height = 0,
    max_cosine_distance = 0.2
    nn_budget = 100

    dummy_run(
        sequence_dirs=sequence_dirs, 
        detection_files=detection_files,
        min_confidence=min_confidence,
        nms_max_overlap=nms_max_overlap,
        min_detection_height=min_detection_height,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget
        )