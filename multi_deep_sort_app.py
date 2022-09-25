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
    assert len(sequence_dirs) == len(detection_files)
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


def run(sequence_dirs, detection_files, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    assert len(sequence_dirs) == len(detection_files)
    n_cams = len(sequence_dirs)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    multi_tracker = MultiTracker(metric, n_cams=n_cams)
    seqs_info = gather_multi_sequence_info(sequence_dirs, detection_files)
    results = n_cams * [[]]
    
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
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

        # Update visualization.
        if display:
            images = [cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
                     for seq_info in seqs_info]
            vis.set_image(images)
            vis.draw_detections(detections)
            vis.draw_trackers(multi_tracker.tracks)

        # Store results.
        for cam_idx in range(n_cams):
            for track in multi_tracker.tracks[cam_idx]:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlwh()
                results[cam_idx].append([
                    frame_idx, track.world_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.MultiVisualization(seqs_info, update_ms=5)
    else:
        visualizer = visualization.NoMultiVisualization(seqs_info)
    visualizer.run(frame_callback)
    


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