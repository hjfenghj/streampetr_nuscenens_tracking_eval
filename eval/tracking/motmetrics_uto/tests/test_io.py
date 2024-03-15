from ...motmetrics_uto import io as Mio
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def test_load_vatic():
    df = Mio.loadtxt(os.path.join(DATA_DIR, 'iotest/vatic.txt'), fmt=Mio.Format.VATIC_TXT)

    expected = pd.DataFrame([
        #F,ID,Y,W,H,L,O,G,F,A1,A2,A3,A4
        (0,0,412,0,430,124,0,0,0,'worker',0,0,0,0),
        (1,0,412,10,430,114,0,0,1,'pc',1,0,1,0),
        (1,1,412,0,430,124,0,0,1,'pc',0,1,0,0),
        (2,2,412,0,430,124,0,0,1,'worker',1,1,0,1)
    ])

    assert (df.reset_index().values == expected.values).all()

def test_load_motchallenge():
    df = Mio.loadtxt(os.path.join(DATA_DIR, 'iotest/motchallenge.txt'), fmt=Mio.Format.MOT15_2D)

    expected = pd.DataFrame([
        (1,1,398,181,121,229,1,-1,-1), #Note -1 on x and y for correcting matlab
        (1,2,281,200,92,184,1,-1,-1),
        (2,2,268,201,87,182,1,-1,-1),
        (2,3,70,150,100,284,1,-1,-1),
        (2,4,199,205,55,137,1,-1,-1),
    ])

    assert (df.reset_index().values == expected.values).all()