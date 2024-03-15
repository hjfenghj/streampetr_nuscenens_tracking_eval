"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
from typing import List, Dict, Callable, Tuple
import unittest

import numpy as np
import sklearn
import tqdm
from pyquaternion import Quaternion

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

import sys
sys.path.append("...")
from eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from eval.tracking.data_classes import TrackingBox, TrackingMetricData
from eval.tracking.mot import MOTAccumulatorCustom
from eval.tracking.render import TrackingRenderer
from eval.tracking.utils import print_threshold_metrics, create_motmetrics

class TrackingEvaluation(object):
    def __init__(self,
                 idsw_dirty_info_file: str,
                 save_idsw_info:bool,
                 save_guanlian_info:bool,
                 save_highlight_info,
                 view_name,
                 min_dist_thr,
                 max_dist_thr,
                 hl_large_dist,
                 hl_small_dist,
                 NUSC,  
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 verbose: bool = True,
                 output_dir: str = None,
                 render_classes: List[str] = None,
                 ):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.metric_worst = metric_worst
        self.verbose = verbose
        self.output_dir = output_dir
        self.render_classes = [] if render_classes is None else render_classes

        self.n_scenes = len(self.tracks_gt)
        self.nusc = NUSC
        self.min_dist_thr = min_dist_thr
        self.max_dist_thr = max_dist_thr
        self.hl_large_dist = hl_large_dist
        self.hl_small_dist = hl_small_dist
        self.view_name = view_name
        self.idsw_dirty_info_file = idsw_dirty_info_file
        self.save_idsw_info = save_idsw_info
        self.save_guanlian_info = save_guanlian_info
        self.save_highlight_info = save_highlight_info

        self.GUANLIAN = {}
        self.HIGHLIGHT = {}
        
        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)

        thresh_metrics = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        mh = create_motmetrics()
        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        thresholds, recalls = self.compute_thresholds(gt_box_count)
        md.confidence = thresholds
        md.recall_hypo = recalls
        if self.verbose:
            print('Computed thresholds\n')

        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes with the exact same score.
            if threshold in thresholds[:t]:
                continue            
            # print("threshold is :{}".format(threshold))
            # Accumulate track data.
            acc, _ = self.accumulate_threshold(threshold)   #  acc_Dict -> {scene_id:{time1:acc,time2:acc}}

            # Compute metrics for current threshold.
            thresh_name = self.name_gen(threshold)
            thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)

            thresh_metrics.append(thresh_summary)

            # print("thresh_summary is:{}".format(thresh_summary.to_dict()))

            # Print metrics to stdout.
            if self.verbose:
                print_threshold_metrics(thresh_summary.to_dict())
        # Concatenate all metrics. We only do this for more convenient access.
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Get the number of thresholds which were not achieved (i.e. nan).
        unachieved_thresholds = np.array([t for t in thresholds if np.isnan(t)])
        num_unachieved_thresholds = len(unachieved_thresholds)

        # Get the number of thresholds which were achieved (i.e. not nan).
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        num_duplicate_thresholds = len(valid_thresholds) - len(np.unique(valid_thresholds))

        # Sanity check.
        assert num_unachieved_thresholds + num_duplicate_thresholds + len(thresh_metrics) == self.num_thresholds

        # Figure out how many times each threshold should be repeated.
        rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]

        # Store all traditional metrics.
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            # Skip metrics which we don't output.
            if metric_name == '':
                continue

            # Retrieve and store values for current metric.
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        worst = np.nan  # We can't know how these error types are distributed.
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once, duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)

            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def compute_thresholds(self, gt_box_count: int) -> Tuple[List[float], List[float]]:
        """curr_class_name
        Compute the score thresholds for predefined recall values.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_box_count: The number scene_tracks_gtof GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # Run accumulate to get the scores of TPs.
        _, scores = self.accumulate_threshold(threshold=None)

        # Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds, [np.nan] * self.num_thresholds

        # Sort scores.
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]

        # Compute recall levels.
        tps = np.array(range(1, len(scores) + 1))
        rec = tps / gt_box_count
        assert len(scores) / gt_box_count <= 1

        # Determine thresholds.
        max_recall_achieved = np.max(rec)
        rec_interp = np.linspace(self.min_recall, 1, self.num_thresholds).round(12)
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # Cast to list.
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # Reverse order for more convenient presentation.
        thresholds.reverse()
        rec_interp.reverse()

        # Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp

    def quaternion_to_rotation_matrix(self,q):  # x, y ,z ,w
        rot_matrix = np.array(
            [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
            [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
            [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
            dtype=q.dtype)
        return rot_matrix

    # 判断某ann在视野外还是视野内
    def view_IO(self,view_name,F):
        if view_name == "ALL":
            sample_token = F.sample_token
            sample = self.nusc.get("sample",sample_token)   #image的token

            cam_sample = sample["data"]["CAM_FRONT"]
            image_info = self.nusc.get('sample_data',cam_sample)                      

            #ego_pose
            ego_data = self.nusc.get("ego_pose",image_info["ego_pose_token"])

            # 标注真值到相机坐标系
            # global frame
            center = np.array(F.ego_translation)
            return True, center
        
        flag = False
        sample_token = F.sample_token
        sample = self.nusc.get("sample",sample_token)   #image的token

        cam_sample = sample["data"][view_name]
        image_info = self.nusc.get('sample_data',cam_sample)
        calib_data = self.nusc.get('calibrated_sensor',image_info['calibrated_sensor_token'])
        image_h = image_info["height"]
        image_w = image_info["width"]                          


        # 标注真值到相机坐标系
        # ego frame
        center = np.array(F.ego_translation)
        center_copy = center
        # 从ego vehicle frame转换到sensor frame
        quaternion = Quaternion(calib_data['rotation']).inverse
        center -= np.array(calib_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)

        intrinsic = calib_data['camera_intrinsic']
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = np.array(intrinsic)
        points = center     #(3,)

        # print(points.shape)
        PP = np.zeros((4,1))
        PP[:3,0] = points
        PP[3,0] = 1

        # points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
        points = np.dot(trans_mat, PP)[:3, :]
        points /= points[2, :]
        points = points.reshape(3,)

        if points[0] < image_w and points[0] >= 0 and points[1] < image_h and points[1] >=0:
            flag = True

        return flag,center_copy


    def accumulate_threshold(self, threshold: float = None) -> Tuple[pandas.DataFrame,  List[float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        accs = []
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.


        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            
            if self.class_name in self.render_classes and threshold is None:
                save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(save_path)
            else:
                renderer = None

            # 每一时刻每一类的gt框和pred框数量
            for timestamp in scene_tracks_gt.keys():

                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]

                frame_gt =   [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]
                    
                # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                if threshold is not None:
                    frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                # Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan   #self.dist_th_tp是距离阈值,一旦超过这个阈值就匹配不上了   # self.dist_th_tp = 2.0

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                _ , guanlian_info, highlight_info = acc.update(threshold, self.class_name, self.hl_large_dist, self.hl_small_dist,\
                                      self.save_idsw_info, self.save_guanlian_info, self.save_highlight_info, \
                                    self.nusc, gt_ids, pred_ids, distances, frameid=frame_id, scene_name = scene_id,\
                                    timestamp=timestamp, idsw_dirty_info_file = self.idsw_dirty_info_file)
                
                if scene_id not in self.GUANLIAN:
                    self.GUANLIAN.setdefault(scene_id,{})
                if threshold not in self.GUANLIAN[scene_id]:
                    self.GUANLIAN[scene_id].setdefault(threshold,{})
                self.GUANLIAN[scene_id][threshold][timestamp] = guanlian_info

                if scene_id not in self.HIGHLIGHT:
                    self.HIGHLIGHT.setdefault(scene_id,{})
                if threshold not in self.HIGHLIGHT[scene_id]:
                    self.HIGHLIGHT[scene_id].setdefault(threshold,{})
                self.HIGHLIGHT[scene_id][threshold][timestamp] = highlight_info

                
                # Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    events = acc.events.loc[frame_id]
                    matches = events[events.Type == 'MATCH']
                    match_ids = matches.HId.values
                    match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                    scores.extend(match_scores)
                else:
                    events = None

                # Render the boxes in this frame.
                if self.class_name in self.render_classes and threshold is None:
                    renderer.render(events, timestamp, frame_gt, frame_pred)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1

            accs.append(acc)

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        return acc_merged, scores


