import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wires_df = pd.read_csv('wires_LUT.txt', delim_whitespace=True)

wh = 1
sec = 10
st = 1
sl = 1

SLayer_1 = wires_df[(wires_df['wheel'] == wh) & (wires_df['sector'] == sec) & (wires_df['station'] == st) & (wires_df['super_layer'] == sl)]

print(SLayer_1)
