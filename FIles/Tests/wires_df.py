import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
wires_df = pd.read_csv('wires_LUT.txt', delim_whitespace=True)

wh = 1
sec = 10
st = 1
sl = 1

SLayer_1 = wires_df[(wires_df['wheel'] == wh) & (wires_df['sector'] == sec) & (wires_df['station'] == st) & (wires_df['super_layer'] == sl)]

SLayer_1['global_r'] = (SLayer_1['global_x']**2 + SLayer_1['global_y']**2)**0.5

print(SLayer_1[SLayer_1['layer'] == 1].head())
print(SLayer_1[SLayer_1['layer'] == 4].head())
