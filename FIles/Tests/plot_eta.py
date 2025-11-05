import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

acceptances_df = pd.read_csv('eta_acceptances.txt', delim_whitespace=True)
st = 'MB1'
eta1_st = 'eta1_' + st
eta2_st = 'eta2_' + st

print(acceptances_df[[eta1_st, eta2_st]])
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.fill_between([-3.2, 3.2], y1=-1.2, y2=acceptances_df[eta2_st].iloc[0], color="limegreen")
# ax.fill_between([-3.2, 3.2], y1=acceptances_df[eta1_st].iloc[0], y2=1.2, color="limegreen")
eta1 = acceptances_df[eta1_st][acceptances_df[eta1_st].notna()].values
eta2 = acceptances_df[eta2_st][acceptances_df[eta2_st].notna()].values
for i in range(0, sum(acceptances_df[eta1_st].notna())):
    print(eta1[i], eta2[i])
    y1 = eta1[i]
    y2 = eta2[i]
    if eta1[i] < -1.2:
        y1 = -1.2
    if eta2[i] > 1.2:
        y2 = 1.2
    ax.fill_between([-3.2, 3.2], y1=y1, y2=y2, color="limegreen")

ax.grid()
fig.savefig("Test_eta.png")
fig.savefig("Test_eta.pdf")
