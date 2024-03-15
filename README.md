# UTO_NU_EVAL

1. 整体说明
该工具箱支持streampetr网络输出的跟踪结果json文件的评估,可以支持不同距离,不同视野,不同类别的跟踪指标输出
该工具箱支持输出关联json,高亮json,IDSW的json文件的输出
关联json(guanlian_info.json): 保存每个场景  每帧数据  预测目标与真值目标的关联信息，用于后续foxglove可视化展现关联线，评估网络的检测性能
ID跳变json(idsw_clear_info.json): 保存最优MOTP(也可以是MOTA,可以自行改动，位置在evaluate_uto.py脚本153行)对应的阈值分数下的ID跳变信息，储存预测目标ID(也可以是真值ID,修改位置在eval/tracking/motmetrics_uto/mot.py的226行)，用于后续配合foxglove分析网络结果
高亮json(highlight_info.json): 保存存在关联关系的预测与真值之间欧式距离超过设置阈值的预测目标ID(也可以是真值ID，修改位置在eval/tracking/motmetrics_uto/mot.py的234行和236行)，用于后续配合focglove分析网络结果
###  注意后续的foxglove可视化仅仅支持预测目标的IDSW高亮和highlight高亮 ###

脚本介绍: evaluate_uto.py   评估脚本,主函数入口

2. 参数说明
main函数参数解读:
result_root         # 用于评估的网络输出的网络跟踪json文件所在的文件夹
output_dir          # 用于储存评估得到的一些图表和曲线等,最好是在评估工具箱内的相对路径
eval_set            # 评估的数据集类型,对应输入json的数据split类型,比如json是v1.0-mini的结果预测,那么这里就也是v1.0-mini
dataroot            # nuscene数据集路径,最好是在这个工具箱内有个原始数据集的软链接路径, 使用相对路径
version             # nuscenes_devite的参数, 可选参数: v1.0-trainval v1.0-mini  v1.0-test
config_path         # nuscenes的评测配置文件, 默认路径为/nuscenes/eval/tracking/configs/tracking_nips_2019.json, 是一些参数(包含距离阈值, 评测指标的默认值, 评测距离等)
render_curves       # 控制是否输出评测指标的曲线
verbose             # 是否运行的时候在终端打印结果
render_classes      # 绘制某个类别的2D图,每个场景一个文件夹，可以动态看出物体的移动
min_dist            # 参与评测的物体距离自车的最近距离, 正方形的区域, 单独计算纵向距离和横向距离
max_dist            # 参与评测的物体距离自车的最远距离, 正方形的区域, 单独计算纵向距离和横向距离
hl_large_dist       # 大目标highlight的距离阈值 "trailer","car", "bus","truck", 预测与真值欧式距离如果超过阈值, 将会输出在highlight.json文件中
hl_small_dist       # 小目标highlight的距离阈值 "bicycle","motorcycle","pedestrian"
view_name           # 评测单视野目标, 可选参数:"CAM_FRONT", "CAM_BACK","CAM_FRONT_RIGHT","CAM_FRONT_LEFT","CAM_BACK_LEFT","CAM_BACK_RIGHT", 并且可以与距离阈值min_dist/maxdist一起使用, 用于评估单视野的不同距离范围目标
save_idsw_info      # 是否需要保存前后帧发生(IDSW)ID跳变的目标ID值, 用于后续可视化ID跳变的目标
save_guanlian_info  # 是否需要保存每一帧预测结果与真实目标的ID关联信息, 用于后续可视化
save_highlight_info # 是否需要保存相互关联的预测与真值目标之间的欧式距离大于阈值的目标ID, 用于后续可视化


3. 文件格式
- 3.1 输出的guanlian_info.json文件格式

"4efbf4c0b77f467385fc2e19da45c989": {                    场景ID
    "1537852775647766": {                                时间戳
        "63c96f4826604052895403f87da8db6c": "2"          真值目标ID : 预测目标ID
    },
    "1537852776147630": {
        "63c96f4826604052895403f87da8db6c": "2"
    },
    "1537852793147806": {
        "5926dc9e8fc8424db74d6f3dd5119942": "16"
    },
    "1537852793647691": {
        "5926dc9e8fc8424db74d6f3dd5119942": "16"
    },
    "1537852794147562": {
        "5926dc9e8fc8424db74d6f3dd5119942": "16"
    },
    "1537852794697749": {}
},

- 3.2 输出的highlight_info.json文件格式
    "696a45dbd11346b794fdce43fa0a1770": {  场景ID
        "1538984909897929": [],            时间戳: highlight的预测预测目标ID
        "1538984910397796": [],
        "1538984919948029": [
            "4",
            "4",
            "4",                           目前的逻辑，会有重复的ID存下来，因为每一个置信度分数阈值都会使用这一个文件，导致的问题。后续可以将不同的阈值分数写为不同的文件名称。
            "5",                           使用目前的结果，也可以支持mcap链路(mcap可视化可以使用set类型的容器使其具有唯一性)
            "4",
            "5",
            "4",
            "7",
            "8",
            "8",
            "7",
            "8",
            "7",
            "8",
            "7",
            "8",
            "7",
            "8",
            "7",
            "8",
            "7",
            "17",
            "1",
            "1",
            "17",
            "1",
            "1",
            "1",
            "1",
            "20",
            "1"
        ]
    }


- 3.3  输出的id跳变json文件格式:
  {
    "scene-0783": {             # 场景ID
        "1535657840549782": [   # 时间戳
            "58"                # 当前帧发生ID跳变的预测目标ID
        ]
    },
    "scene-0098": {
        "1533151366948018": [
            "24"
        ]
    },
    "scene-0345": {
        "1533201591549843": [
            "72"
        ]
    },
    "scene-0099": {
        "1533151420697955": [
            "39"
        ]
    },
    "scene-0910": {
        "1537298069798403": [
            "22"
        ]
    },
    "scene-0904": {
        "1537297946198546": [
            "12"
        ]
    },
    "scene-0557": {
        "1535730290047984": [
            "68"
        ]
    }
  }



3.4 不同评估指标的意义:


MOTP         衡量了多目标跟踪位置误差的指标                                           lower            如果位置距离使用欧式距离，指标越小越好。如果使用IOU的重叠度，那就是越大越好
MOTA         衡量多目标跟踪准确度的一个指标                                           higher
MOTAR        关注目标的距离估计，以评估系统在测量目标距离时的准确性                       higher
MT           大多数跟踪,一条轨迹被跟踪到80%以上就认为是MT                              higher
ML           大部分缺失跟踪                                                        lower
IDS          pred_ID跳变数量                                                      lower
FRAG         轨迹碎片化的总次数,一条轨迹被切断的次数                                   lower
TID          目标跟踪算法在多个时间步骤中维护跟踪对象的身份的能力                        higher
LGD          旨在衡量跟踪器对目标位置估计的质量                                       lower
FAF          每帧的平均误报警数                                                     lower
AMOTP        所有阈值分数求出的MOTP的均值(若某阈值分数下,motp为nan,使用默认参数2.0)      lower
AMOTA        所有阈值分数求出的MOTAR的均值(若某阈值分数下,motp为nan,使用默认参数0)       higher

IDSW的计算过程: 维护一个匹配容器，通过距离矩阵，使用匈牙利算法完成预测目标与真值目标的分配，在容器中记录真值ID和预测ID的对应关系。另外还要记录每个真值ID出现的最后一帧的帧数;
              发生IDSW的预测目标需要满足三个条件:   首先记历史匹配容器为M, 当前帧的某真值目标为T, 与当前帧T对应的预测目标为P, 当前帧帧号(表示时间信息)为H1, 目标T出现的上一帧帧号为H
                                            i.    通过匈牙利算法求出的 真值目标T  需要存在于匹配容器M中(防止第一次出现的目标P被记录成IDSW目标)
                                            ii.   历史匹配容器M中, 与T对应的预测目标为P1, 且P1 != P(历史出现过的真值目标，前后对应的预测目标ID不一致)
                                            iii.  H1 - H < MAX_SWITCH_TIME(设置的跳变时间阈值)(为了避免将目标跟踪中的短暂丢失判定为IDSW)
                                            这种思路可以过滤掉短暂的目标丢失或者重新出现，从而更准确地评估跟踪器的性能。
                                            比如说某帧真值目标A对应的是预测目标P2, 历史匹配容器中这个真值ID对应的是预测目标P3, 但是因为这个真值目标已经很久没有出现了，所以不记为IDSW

                                            帧数差阈值MAX_SWITCH_TIME的高低说明什么问题？
                                            答: 设置帧数差的阈值较高意味着对于目标在不同帧之间出现的间隔较大的情况更加宽容。这样会导致跟踪器对于目标在一段时间内消失然后重新出现的情况不会过分惩罚，可能会将这种情况视为同一目标的跟踪而不是ID切换。
                                                设置帧数差的阈值较低意味着对于目标在不同帧之间出现的间隔较小的情况更加严格。这样会导致跟踪器对于目标在一段时间内消失然后重新出现的情况进行更严格的判断，可能会减少误判，但也可能会增加因为真实的ID切换而导致的IDSW值

                                            这个工具箱只是记录了发生ID切换的个数, 没有记录发生ID merges(合并)和ID splits(拆分)的情况
                                            ID merges: 同一个真实目标在不同的帧中被错误地标识为不同的跟踪标识(ID),
                                                       假设有一辆汽车在视频中连续出现，但由于视角变化或者遮挡等因素，跟踪器在不同的帧中将同一辆车误认为不同的ID。例如，在第一帧中将车辆标识为ID1，然后在下一帧中将同一辆车误认为新的ID2，这就是ID拆分的情况
                                            ID splits: 同一帧中，多个不同的真实目标被错误的标识为同一个跟踪标识(ID)
                                                       假设有两辆相似的汽车在视频中同时出现，但由于它们外观相似，跟踪器错误地将它们标识为同一个ID。例如，在第一帧中将第一辆车标识为ID1，然后在下一帧中将第二辆车错误地标识为相同的ID1，这就是ID合并的情况


MISS值(FN, 漏检)的计算过程: MISS值表示当前帧的真值目标，没有与之匹配的预测目标(通过匈牙利匹配),也就是说漏检查的目标个数
FP(误检）值的计算过程  : FP值表示当前帧的预测目标没有与之对应的真值目标，也就是说误检的个数
FAF(假警报频率, False Alarm Frequency)的计算过程   : FP / 总帧数
LGD(最长间隔持续时间, Longest Gap Duration): 记录所有发生间隔的目标的间隔帧数和 / 所有被跟踪到的目标(在整个视频序列中, 只要目标在任意一帧被跟踪到，并且有一个对应的预测目标与之对应，就会被计数)
TID(Track ID)评估目标跟踪算法在跟踪过程中目标的持续性和一致性: 每一个跟踪的目标的连续出现帧数/总出现帧数为一个值，将所有被跟踪到的目标的结果相加  除于被跟踪到的目标个数。 
                                                       注意同一个目标在序列中的两段连续检出被记为两个不同的片段，跟踪目标也记为两个。这样也可以一定程度上惩罚中间漏检情况

FRAG的计算过程:  就说每具体比较这指标，没有研究过。应该是所有被跟踪到的目标个数，每一个被跟踪到的目标记为一个完整的跟踪段落。这个段落在整个序列中因为中间帧漏检的情况，分离为多个断裂的片段
                所有被跟踪到的目标的所有断裂片段总和 除于 总的被跟踪到的目标个数  

MOTA的计算过程:  1 - (IDSW + FP + FN) / 总GT数量   IDSW, FP和FN均为该split(trainval, test或者mini)下所有的场景下所有序列的累积，总GT表示所有场景所有序列的GT数总和。  每个置信度阈值分数对应一个MOTA, 使用最大MOTA对应的阈值分数
MOTP的计算过程:  一个目标在整个序列的100帧，其中有60帧有预测目标与之匹配，每一次匹配都有一个距离差值, 将所有被跟踪目标(就是在整个序列中有过与之匹配的预测目标，标识这个真值目标我们跟踪到了)的所有的距离差值求和。除于所有被跟踪目标在所有序
               列中的所有匹配次数和，得到MOTP.      一个阈值对应一个MOTP，可与MOTA同样操作

AMOTA和AMOTP:  在这个工具箱中，是所有置信度阈值分数(也可以说成所有的RECALL,因为一个置信度分数可以求出一个RECALL, 阈值分数的bin数划分也是根据RECALL来划分的。比如说0.1.0.2,...,1.0的RECALL对应10个不同的置信度阈值分数)对应的MOTA和MOTP的均值
MOTAR的计算方式: MOTA/MOTP


简称：全称
MOTAR: Multiple Object Tracking Accuracy Ratio
MOTP: Multiple Object Tracking Precision
MOTA: Multiple Object Tracking Accuracy
LGD: Longest Gap Duration
FAF: False Alarm Frequency






1. floder_tree

data/
├── can_bus
├── maps
├── samples
├── sweeps
├── v1.0-mini
├── v1.0-test
└── v1.0-trainval

tracking_results/
└── tracking_result.jso

output/
├── eval_dirs
│   ├── ALL_0.0_10.0_metrics_details.json           # ALL代表全视野， 0.0_10.0 表示评估距离为0m到10m
│   └── ALL_0.0_10.0_metrics_summary.json
├── guanlian_info.json                              # 关联线信息json文件                   
├── highlight_info.json                             # 预测真值关联目标欧式距离超过阈值的高亮目标json文件
├── idsw_clear_info.json                            # 相比上一帧的某个目标A, 在当前帧中这个目标发生了ID跳变，保存当前帧目标A的预测ID(用于后续的foxglove可视化)
├── idsw_dirty_info.json                            # 相比上一帧的某个目标A, 在当前帧中这个目标发生了ID跳变，保存当前帧目标A的预测ID(用于后续的foxglove可视化)，dirty是记录了每个阈值分数下的结果在一个json中，然后通过最优阈值分数提出对应的部分
└── plots                                           # 储存各评估指标的曲线图
    ├── faf.pdf
    ├── fn.pdf
    ├── fp.pdf
    ├── frag.pdf
    ├── ids.pdf
    ├── lgd.pdf
    ├── ml.pdf
    ├── mota.pdf
    ├── motp.pdf
    ├── mt.pdf
    ├── summary.pdf
    ├── tid.pdf
    └── tp.pdf

5. 依赖环境   
   python3.8

cachetools==5.3.3
contourpy==1.1.1
cycler==0.12.1
fonttools==4.49.0
importlib_resources==6.3.0
joblib==1.3.2
kiwisolver==1.4.5
matplotlib==3.7.5
numpy==1.24.4
opencv-python==4.9.0.80
packaging==24.0
pandas==1.4.4
pillow==10.2.0
pyparsing==3.1.2
pyquaternion==0.9.9
python-dateutil==2.9.0.post0
pytz==2024.1
scikit-learn==1.3.2
scipy==1.10.1
six==1.16.0
threadpoolctl==3.3.0
tqdm==4.66.2
tzdata==2024.1
zipp==3.18.0