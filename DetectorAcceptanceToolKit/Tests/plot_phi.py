import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acceptances_df = pd.read_csv('phi_acceptances.txt', delim_whitespace=True)
st = 'MB1'
phi1_st = 'phi1_' + st
phi2_st = 'phi2_' + st

print(acceptances_df[[phi1_st, phi2_st]])
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.fill_between([-3.2, acceptances_df[phi2_st].iloc[6]], y1=0.0, y2=1.0, color="limegreen")
ax.fill_between([acceptances_df[phi1_st].iloc[6], 3.2], y1=0.0, y2=1.0, color="limegreen")
for i in range(sum(acceptances_df[phi1_st].notna())):
    phi1 = acceptances_df[phi1_st][acceptances_df[phi1_st].notna()].values
    phi2 = acceptances_df[phi2_st][acceptances_df[phi2_st].notna()].values
    if i == 6: continue
    ax.fill_between([phi1[i], phi2[i]], y1=0.0, y2=1.0, color="limegreen")

ax.grid()
fig.savefig("Test_kind_0.png")
fig.savefig("Test_kind_0.pdf")
