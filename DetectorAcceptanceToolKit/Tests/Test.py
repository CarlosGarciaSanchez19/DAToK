import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wires_df = pd.read_csv('wires_LUT.txt', delim_whitespace=True)

def chamber_range(wh, sec, st, sl=1):
    Slayer_df = wires_df[(wires_df['wheel'] == wh) & (wires_df['sector'] == sec) & (wires_df['station'] == st) & (wires_df['super_layer'] == sl)]
    if Slayer_df.empty:
        print(f'No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}')
        return None, None
    num_layers = int(Slayer_df['layer'].max())

    min_diff = 1000
    phi1 = -5
    phi2 = 5
    if sec == 7:
        phi1, phi2 = Slayer_df[Slayer_df['phi'] > 0]['phi'].min(), Slayer_df[Slayer_df['phi'] < 0]['phi'].max()
    elif phi1 < 0 and phi2 < 0:
        phi1, phi2 = Slayer_df['phi'].max(), Slayer_df['phi'].min()
    else:
        phi1, phi2 = Slayer_df['phi'].min(), Slayer_df['phi'].max()
    return phi1, phi2

# max_wh = wires_df['wheel'].max()
# max_sec = wires_df['sector'].max()
# max_st = wires_df['station'].max()
min_wh = 2
max_wh = 0

min_sec = 0
max_sec = wires_df['sector'].max()

min_st = 0
max_st = wires_df['station'].max()

acceptance = np.zeros((max_wh * 2 + 1, max_sec, max_st, 2))
# phi1 = []
# phi2 = []

for wh in range(-2 + min_wh, max_wh + 1):
    for sec in range(1 + min_sec, max_sec + 1):
        for st in range(1 + min_st, max_st + 1):
            # print(f'Wheel: {wh}, Sector: {sec}, Station: {st}')
            phi1_st, phi2_st =  chamber_range(wh, sec, st)
            # print(f'Phi1: {phi1_st}, Phi2: {phi2_st}')
            acceptance[wh, sec - 1, st - 1] = [phi1_st, phi2_st]
            # phi1.append(phi1_st)
            # phi2.append(phi2_st)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for st in range(1 + min_st, max_st + 1):
    phi1 = acceptance[0, :, st - 1, 0]
    phi2 = acceptance[0, :, st - 1, 1]

    phi1 = phi1[~(phi1 == None)]
    phi2 = phi2[~(phi2 == None)]
    x_phi1 = st * np.cos(phi1)
    y_phi1 = st * np.sin(phi1)
    x_phi2 = st * np.cos(phi2)
    y_phi2 = st * np.sin(phi2)

    start_points = np.array([[x_phi1, y_phi1]])
    end_points = np.array([[x_phi2, y_phi2]])

    label1 = None
    label2 = None
    if st == 1 + min_st:
        label1 = 'Phi1'
        label2 = 'Phi2'
    ax.scatter(start_points[:, 0], start_points[:, 1], color='blue', label=label1)
    ax.scatter(end_points[:, 0], end_points[:, 1], color='red', label=label2)

    for i, (start, end) in enumerate(zip(start_points, end_points)):
        x_values = [start[0], end[0]]
        y_values = [start[1], end[1]]
        ax.plot(x_values, y_values, c='k', linestyle='-')

# plt.plot(x_phi1[neg1], y_phi1[neg1], 'r+', label='Phi1')
# plt.plot(x_phi1[~neg1], y_phi1[~neg1], 'b+', label='Phi1')
# plt.plot(x_phi2[neg2], y_phi2[neg2], 'r*', label='Phi2')
# plt.plot(x_phi2[~neg2], y_phi2[~neg2], 'b*', label='Phi2')

# plt.plot(x_phi1, y_phi1, 'r+', label='Phi1', markersize=2)
# plt.plot(x_phi2, y_phi2, 'b*', label='Phi2')

# plt.scatter(x_phi1, y_phi1, c='r', label='Phi1')
# plt.scatter(x_phi2, y_phi2, c='b', label='Phi2')


ax.lines[2].set_label('Chambers_max_range')
ax.legend()
ax.grid()
plt.savefig('chamber_ranges.png')
plt.savefig('chamber_ranges.pdf')
