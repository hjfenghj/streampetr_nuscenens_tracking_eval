# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

import numpy as np
from nuscenes_uto import NuScenes
from eval.common.config import config_factory
from eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from eval.tracking.algo import TrackingEvaluation
from eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
     TrackingMetricData
from eval.tracking.loaders import create_tracks
from eval.tracking.render import recall_metric_curve, summary_plot
from eval.tracking.utils import print_final_metrics

class TrackingEval:
    def __init__(self,
                 save_gulian_info,
                 guanlian_file,
                 highlight_file,
                 idsw_dirty_info_file: str,
                 idsw_clear_info_file: str,
                 save_idsw_info: bool,
                 save_highlight_info,
                 view_name,
                 min_Dist,
                 max_Dist,
                 hl_large_dist,
                 hl_small_dist,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None,
                 ):
        self.cfg = config
        
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.render_classes = render_classes
        self.min_dist_thr = min_Dist
        self.max_dist_thr = max_Dist
        self.hl_large_dist = hl_large_dist
        self.hl_small_dist = hl_small_dist
        self.view_name = view_name
        self.idsw_dirty_info_file = idsw_dirty_info_file
        self.idsw_clear_info_file = idsw_clear_info_file
        self.save_idsw_info = save_idsw_info
        self.save_highlight_info = save_highlight_info

        self.save_gulian_info = save_gulian_info
        self.guanlian_file = guanlian_file
        self.highlight_file = highlight_file
        # Check result file exists.
        assert os.path.exists(self.result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Initialize NuScenes object.
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)
        self.nusc = nusc

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose)
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, self.view_name, pred_boxes, self.min_dist_thr, self.max_dist_thr, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, self.view_name, gt_boxes, self.min_dist_thr, self.max_dist_thr, self.cfg.class_range, verbose=verbose)
        self.sample_tokens = gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)
        
    def read_idsw_file(self,idsw_dirty_file):
        text = []
        with open(idsw_dirty_file,'r') as fp:
            for line in fp:
                line = line.split("\n")
                text.append(line[0].split(" "*20))

        return np.array(text)


    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        classes_infos = {}
        highglight_infos = {}
        def accumulate_class(curr_class_name):
            curr_ev = TrackingEvaluation(self.idsw_dirty_info_file,self.save_idsw_info,self.save_gulian_info,self.save_highlight_info,self.view_name,\
                                         self.min_dist_thr,self.max_dist_thr,self.hl_large_dist,self.hl_small_dist,self.nusc,\
                                         self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,self.cfg.dist_th_tp, \
                                         self.cfg.min_recall,num_thresholds=TrackingMetricData.nelem,metric_worst=self.cfg.metric_worst, \
                                         verbose=self.verbose,output_dir=self.output_dir,render_classes=self.render_classes,
                                         )
            
            curr_md = curr_ev.accumulate()   

            classes_infos[curr_class_name] = curr_ev.GUANLIAN
            highglight_infos[curr_class_name] = curr_ev.HIGHLIGHT

            metric_data_list.set(curr_class_name, curr_md)

        for class_name in self.cfg.class_names:
            """
            bicycle/bus/car/motorcycle/pedestrian/trailer/truck
            """
            accumulate_class(class_name)

        # -----------------------------------
        # Step 2: Aggregate metrics from the metric data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        target_dict = {}
        best_score_dict = {}
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional metrics.
            # If multiple thresholds have the same value, pick the one with the highest recall.
            md = metric_data_list[class_name]  
            if np.all(np.isnan(md.mota)):   # len(md.motp) = 40
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            if best_thresh_idx is not None:
                best_score_dict[class_name] = md.confidence[best_thresh_idx]

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            # 这里的AMOTP仅仅是指40个阈值分数对应的motp的均值，如果某阈值分数下的motp为nan的时候,就用配置配置文件tracking_nip_2019.json中的metric_wrost->amotp=2.0替代, 注意与AMOTP概念的不同
            # motp计算了所有帧(某数据分支,val/train/test)的motp
            # 这里motp就计算了所有scene中的预测结果，没有分scene求对应的motp，然后再求所有scene的amotp
            # amota与amotp求解一致
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

            if best_thresh_idx is not None and self.save_idsw_info:   
                if not os.path.getsize(self.idsw_dirty_info_file):
                    continue
                source_texts = self.read_idsw_file(self.idsw_dirty_info_file)
                target_texts = source_texts[(source_texts[:,3] == str(md.confidence[best_thresh_idx])) & (source_texts[:,4] == class_name)]
                
                for target_text in target_texts:
                    if target_text[0] not in target_dict:
                        target_dict.setdefault(target_text[0],{})
                    if target_text[1] not in target_dict[target_text[0]]:
                        target_dict[target_text[0]].setdefault(target_text[1],[])
                    target_dict[target_text[0]][target_text[1]].append(target_text[2])

        
        if self.save_idsw_info:
            json_str = json.dumps(target_dict, indent=4)
            with open(self.idsw_clear_info_file,'w') as fp:
                fp.write(json_str) 
        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        ### 保存每个scene中的高亮目标ID(预测与真值欧式距离超过阈值)为json文件，配合后续可视化
        if self.save_highlight_info:
            highlight_dict = {}
            for scene_name in self.tracks_gt.keys():
                if scene_name not in highlight_dict:
                    highlight_dict.setdefault(scene_name,{})
                for class_name in self.cfg.class_names:
                    if scene_name not in highglight_infos[class_name]:
                        continue
                    # 表示这个场景中某物体缺失，所以没有这个阈值分数的内容
                    if best_score_dict[class_name] not in highglight_infos[class_name][scene_name]:
                        continue

                    times_acc = highglight_infos[class_name][scene_name][best_score_dict[class_name]]
                    #   将该场景下的不同时刻对应的sample的关联储存
                    for tk,tv in times_acc.items():          #tk -> 时间戳信息
                        if tk not in highlight_dict[scene_name]:
                            highlight_dict[scene_name].setdefault(tk,[])
                        for info in tv:
                            highlight_dict[scene_name][tk].append(info[1])         #记录该时间戳下的pred高亮目标 
                json_str = json.dumps(highlight_dict,indent=4)
                with open(self.highlight_file,'w') as fp:
                    fp.write(json_str)  

        ### 保存每个scene中每一帧预测结果与真值的关联信息，配合后续可视化
        if self.save_gulian_info:
            guanlian_dict = {}
            for scene_name in self.tracks_gt.keys():
                if scene_name not in guanlian_dict:
                    guanlian_dict.setdefault(scene_name,{})
                for class_name in self.cfg.class_names:
                    if scene_name not in classes_infos[class_name]:
                        continue
                    # 表示这个场景中某物体缺失，所以没有这个阈值分数的内容
                    if best_score_dict[class_name] not in classes_infos[class_name][scene_name]:
                        continue

                    times_acc = classes_infos[class_name][scene_name][best_score_dict[class_name]]
                    #   将该场景下的不同时刻对应的sample的关联储存
                    for tk,tv in times_acc.items():          #tk -> 时间戳信息
                        if tk not in guanlian_dict[scene_name]:
                            guanlian_dict[scene_name].setdefault(tk,{})

                        for info in tv:
                            guanlian_dict[scene_name][tk][info[0]] = info[1]   # info[0]是gt的id  info[1]是pred的id     应该是int  需要与foxglove的时候的类型对应

                json_str = json.dumps(guanlian_dict,indent=4)
                with open(self.guanlian_file,'w') as fp:
                    fp.write(json_str)
                  

        return metrics, metric_data_list

    def render(self, md_list: TrackingMetricDataList) -> None:
        """
        Renders a plot for each class and each metric.
        :param md_list: TrackingMetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # Plot a summary.
        summary_plot(self.cfg, md_list, savepath=savepath('summary'))

        # For each metric, plot all the classes in one diagram.
        for metric_name in LEGACY_METRICS:
            recall_metric_curve(self.cfg, md_list, metric_name, savepath=savepath('%s' % metric_name))

    def main(self, render_curves: bool = False) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The serialized TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:  # true
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        # metrics_summary['meta'] = self.meta.copy()
        
        eval_result_dir = os.path.join(self.output_dir,"eval_dirs")
        if not os.path.exists(eval_result_dir):
            os.makedirs(eval_result_dir)
        with open(os.path.join(eval_result_dir, self.view_name + "_" + str(self.min_dist_thr) + "_" + str(self.max_dist_thr) + '_metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(eval_result_dir, self.view_name + "_" +  str(self.min_dist_thr) + "_" + str(self.max_dist_thr) + '_metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print metrics to stdout.
        if self.verbose:
            print_final_metrics(metrics)

        # Render curves.
        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 用于评测stresmpetr的tracking结果(json文件)

    parser.add_argument('--result_root', type=str,default="./tracking_results")       # 用于评估的网络输出的网络跟踪json文件
    parser.add_argument('--output_dir', type=str, default='./output')                 # 用于储存评估得到的一些图表和曲线等,最好也是在评估工具箱内的相对路径
    parser.add_argument('--eval_set', type=str, default='val')                        # 评估的数据集类型,对应输入json的数据split类型,比如json是v1.0-mini的结果预测,那么这里就也是v1.0-mini
    parser.add_argument('--dataroot', type=str, default='./data')                     # nuscene数据集路径,最好是在这个工具箱内有个原始数据集的软链接路径, 使用相对路径
    parser.add_argument('--version', type=str, default='v1.0-trainval')               # nuscenes_devite的参数, 可选参数: v1.0-trainval v1.0-mini  v1.0-test
    parser.add_argument('--config_path', type=str, default='')                        # nuscenes的评测配置文件, 默认路径为/nuscenes/eval/tracking/configs/tracking_nips_2019.json, 是一些参数(包含距离阈值, 评测指标的默认值, 评测距离等)
    parser.add_argument('--render_curves', type=int, default=1)                       # 控制是否输出评测指标的曲线
    parser.add_argument('--verbose', type=int, default=1)                             # 是否运行的时候在终端打印结果
    parser.add_argument('--render_classes', type=str, default='', nargs='+', help='For which classes we render tracking results to disk.')  # 绘制某个类别的2D图,每个场景一个文件夹，可以动态看出物体的移动
    parser.add_argument("--min_dist",type=float, default=0.0)                          # 参与评测的物体距离自车的最近距离, 正方形的区域, 单独计算纵向距离和横向距离
    parser.add_argument("--max_dist",type=float, default=10.0)                         # 参与评测的物体距离自车的最远距离, 正方形的区域, 单独计算纵向距离和横向距离
                                                                                      # 最后的状态是个回字形状
    parser.add_argument("--hl_large_dist", type=float, default=1.0)                     # 大目标highlight的距离阈值 "trailer","car", "bus","truck", 预测与真值欧式距离如果超过阈值, 将会输出在highlight.json文件中
    parser.add_argument("--hl_small_dist", type=float, default=0.5)                     # 小目标highlight的距离阈值 "bicycle","motorcycle","pedestrian"
    parser.add_argument("--view_name", type=str, default="ALL")                         # 评测单视野目标, 可选参数:"CAM_FRONT", "CAM_BACK","CAM_FRONT_RIGHT","CAM_FRONT_LEFT","CAM_BACK_LEFT","CAM_BACK_RIGHT", 
                                                                                      # 并且可以与距离阈值min_dist/maxdist一起使用, 用于评估单视野的不同距离范围目标
    parser.add_argument("--save_idsw_info", type = bool, default= True)                # 是否需要保存前后帧发生(IDSW)ID跳变的目标ID值, 用于后续可视化ID跳变的目标
    parser.add_argument("--save_guanlian_info", type=bool, default = True)             # 是否需要保存每一帧预测结果与真实目标的ID关联信息, 用于后续可视化
    parser.add_argument("--save_highlight_info", type=bool, default = True)            # 是否需要保存相互关联的预测与真值目标之间的欧式距离大于阈值的目标ID, 用于后续可视化

    args = parser.parse_args()

    output_dir_ = os.path.expanduser(args.output_dir)

    if args.config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(args.config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialtrackize(json.load(_f))

    idsw_dirty_info_file = os.path.join(output_dir_, "idsw_dirty_info.json")
    idsw_clear_info_file= os.path.join(output_dir_, "idsw_clear_info.json")    
    guanlian_file = os.path.join(output_dir_, "guanlian_info.json")
    highlight_file = os.path.join(output_dir_, "highlight_info.json")
    result_path_ = os.path.join(args.result_root, "tracking_result.json")
    
    nusc_eval = TrackingEval(args.save_guanlian_info, guanlian_file, highlight_file,idsw_dirty_info_file, idsw_clear_info_file, args.save_idsw_info,
                             args.save_highlight_info, args.view_name, args.min_dist, args.max_dist, args.hl_large_dist, args.hl_small_dist,
                             config=cfg_, result_path=result_path_, eval_set=args.eval_set, output_dir=output_dir_, nusc_version=args.version, 
                             nusc_dataroot=args.dataroot, verbose=bool(args.verbose), render_classes=args.render_classes)
    
    nusc_eval.main(render_curves=bool(args.render_curves))
