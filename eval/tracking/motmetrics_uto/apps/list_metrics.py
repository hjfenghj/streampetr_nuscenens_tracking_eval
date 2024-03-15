"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

if __name__ == '__main__':
    from ...motmetrics_uto import metrics as Mmetrics
    
    mh = Mmetrics.metrics.create()
    print(mh.list_metrics_markdown())