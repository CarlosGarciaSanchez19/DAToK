import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

eta_acceptances_df = pd.read_csv('eta_acceptances.txt', delim_whitespace=True)
phi_acceptances_df = pd.read_csv('phi_acceptances.txt', delim_whitespace=True)

st = 'MB1'
eta1_st = 'eta1_' + st
eta2_st = 'eta2_' + st
phi1_st = 'phi1_' + st
phi2_st = 'phi2_' + st


print(eta_acceptances_df[[eta1_st, eta2_st]])
print(phi_acceptances_df[[phi1_st, phi2_st]])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
eta1 = eta_acceptances_df[eta1_st][eta_acceptances_df[eta1_st].notna()].values
eta2 = eta_acceptances_df[eta2_st][eta_acceptances_df[eta2_st].notna()].values
phi1 = phi_acceptances_df[phi1_st][phi_acceptances_df[phi1_st].notna()].values
phi2 = phi_acceptances_df[phi2_st][phi_acceptances_df[phi2_st].notna()].values
for i in range(len(eta1)):
    y1 = eta1[i]
    y2 = eta2[i]
    if eta1[i] < -1.2:
        y1 = -1.2
    if eta2[i] > 1.2:
        y2 = 1.2
    ax.fill_between([phi_acceptances_df[phi1_st].iloc[6], 3.2], y1=y1, y2=y2, color="limegreen")
    ax.fill_between([-3.2, phi_acceptances_df[phi2_st].iloc[6]], y1=y1, y2=y2, color="limegreen")
    for j in range(len(phi1)):
        if j == 6: continue
        ax.fill_between([phi1[j], phi2[j]], y1=y1, y2=y2, color="limegreen")

ax.grid()
fig.savefig("eta_phi_map.png")
fig.savefig("eta_phi_map.pdf")
