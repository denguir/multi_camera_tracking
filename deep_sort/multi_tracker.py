from deep_sort import nn_matching
from .tracker import Tracker
from . import linear_assignment


class MultiTracker:
    """
    This is the multi-camera multi-target tracker.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, n_cams=1):
        assert n_cams >= 1, "Need at least one cam"
        self._id = 0
        self.metric = metric
        self.max_age = max_age
        self.n_cams = n_cams
        self.max_cosine_distance = 1.0
        self.trackers = [Tracker(metric, max_iou_distance, max_age, n_init) for _ in range(n_cams)]
    
    @property
    def tracks(self):
        tr = []
        for tracker in self.trackers:
            tr += [track for track in tracker.tracks if track.world_id is not None]
        return tr

    @property
    def next_id(self):
        self._id += 1
        return self._id

    def predict(self):
        for tracker in self.trackers:
            tracker.predict()
    
    def update(self, detections):
        """
        detetions: List[List[deep_sort.detection.Detection]]
        """
        for i, tracker in enumerate(self.trackers):
            tracker.update(detections[i])

        self._update_track_ids()

    def _update_track_ids(self):
        if self.n_cams == 1:
            tracker = self.trackers[0]
            for track in tracker.tracks:
                if not track.world_id:
                    track.world_id = self.next_id

        for i in range(self.n_cams):
            for j in range(i+1, self.n_cams):
                matches, unmatched_tracks_query, unmatched_tracks_target = self._match(self.trackers[i], self.trackers[j])
                for id_query, id_target in matches:
                    track_query = self.trackers[i].tracks[id_query]
                    track_target = self.trackers[j].tracks[id_target]

                    if not track_query.world_id:
                        track_query.world_id = track_target.world_id
                    elif not track_target.world_id:
                        track_target.world_id = track_query.world_id
                    
                    if track_query.world_id and (track_query.world_id != track_target.world_id):
                        world_id = self.next_id
                        track_query.world_id = world_id
                        track_target.world_id = world_id
                
                for id_query in unmatched_tracks_query:
                    track_query = self.trackers[i].tracks[id_query]
                    track_query.world_id = self.next_id

                for id_target in unmatched_tracks_target:
                    track_target = self.trackers[j].tracks[id_target]
                    track_target.world_id = self.next_id

    def _match(self, tracker_query, tracker_target):

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks_query = [
            i for i, t in enumerate(tracker_query.tracks) if t.is_confirmed()]

        confirmed_tracks_target = [
            i for i, t in enumerate(tracker_target.tracks) if t.is_confirmed()]

        matches, unmatched_tracks_query, unmatched_tracks_target = \
                linear_assignment.min_cost_matching(
                    nn_matching.nn_cost, self.max_cosine_distance, tracker_query.tracks,
                    tracker_target.tracks, confirmed_tracks_query, confirmed_tracks_target)

        return matches, unmatched_tracks_query, unmatched_tracks_target