import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.chambers_phi_acc import Ch_phi_acc
from src.chambers_eta_acc import Ch_eta_acc
from src.chambers_phi_eta_acc import Ch_phi_eta_acc

# cpa = Ch_phi_acc(False)
# cea = Ch_eta_acc(True)

parser = argparse.ArgumentParser(description="Detector Acceptance Tool Kit")
parser.add_argument("-v", "--verbose", action="count", default=0,
                    help="Sets the verbosity. Without it only saving messages are shown. If included, extra information is shown.")
parser.add_argument("-p", "--plot", action="store_true",
                    help="If included, maps are created. If not, acceptances are computed and saved as .npy files.")

args = parser.parse_args()

if args.plot:
    cpea = Ch_phi_eta_acc(args.verbose, eta_acc_file="eta_acceptances.npy", phi_acc_file="phi_acceptances.npy")
    for st in range(cpea.min_st, cpea.max_st + 1):
        cpea.plot2D_map(st=st)
else:
    cpea = Ch_phi_eta_acc(args.verbose)
    cpea.save_eta_acceptances_as_np_obj()
    cpea.save_phi_acceptances_as_np_obj()

# cpea.save_eta_acceptances_to_txt()
# cpea.save_phi_acceptances_to_txt()

# ranges, acceptances = cpa.compute_phi_acceptance(acc=True, rang=rang)

# cea.compute_eta_acceptance()

# cpa.compute_phi_acceptance(acc=True, rang=False)

# cea.save_acceptances_to_txt()
# cpa.save_acceptances_to_txt()

# cpa.plot_ch_phi_acc(acc=True, rang=False, kind=1, outfile="chambers_acceptance_type0")

# ranges = cpa.ranges
# acceptances = cpa.acceptances

# wires_df = pd.read_csv('wires_LUT.txt', delim_whitespace=True)

# def chamber_phi_range(wh, sec, st, sl=1):
#     Slayer_df = wires_df[(wires_df['wheel'] == wh) & (wires_df['sector'] == sec) & (wires_df['station'] == st) & (wires_df['super_layer'] == sl)]
#     if Slayer_df.empty:
#         print(f'No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}')
#         return None, None
#     num_layers = int(Slayer_df['layer'].max())

#     min_diff = 1000
#     phi1 = -5
#     phi2 = 5
#     if sec == 7:
#         phi1, phi2 = Slayer_df[Slayer_df['phi'] > 0]['phi'].min(), Slayer_df[Slayer_df['phi'] < 0]['phi'].max()
#     elif phi1 < 0 and phi2 < 0:
#         phi1, phi2 = Slayer_df['phi'].max(), Slayer_df['phi'].min()
#     else:
#         phi1, phi2 = Slayer_df['phi'].min(), Slayer_df['phi'].max()
#     return phi1, phi2

# def chamber_phi_acceptance(wh, sec, st, sl=1):
#     Slayer_df = wires_df[(wires_df['wheel'] == wh) & (wires_df['sector'] == sec) & (wires_df['station'] == st) & (wires_df['super_layer'] == sl)]
#     if Slayer_df.empty:
#         print(f'No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}')
#         return None, None
#     num_layers = int(Slayer_df['layer'].max())

#     min_diff = 1000
#     phi1 = -5
#     phi2 = 5
#     for i in range(1, num_layers + 1):
#         i_layer_df = Slayer_df[Slayer_df['layer'] == i]
#         if sec == 7:
#             min_phi_layer = i_layer_df[i_layer_df['phi'] > 0]['phi'].min()
#         else:
#             min_phi_layer = i_layer_df['phi'].min()

#         for j in range(1, num_layers + 1):
#             j_layer_df = Slayer_df[Slayer_df['layer'] == j]
#             if sec == 7:
#                 max_phi_layer = j_layer_df[j_layer_df['phi'] < 0]['phi'].max()
#             else:
#                 max_phi_layer = j_layer_df['phi'].max()
#             diff = abs(max_phi_layer - min_phi_layer)
#             if diff < min_diff:
#                 min_diff = diff
#                 phi1 = min_phi_layer
#                 phi2 = max_phi_layer
#     return phi1, phi2

# # min_wh = wires_df['wheel'].max()
# # min_sec = wires_df['sector'].max()
# # min_st =  wires_df['station'].max()

# min_wh = wires_df['wheel'].min()
# max_wh = wires_df['wheel'].max()

# min_sec = wires_df['sector'].min()
# max_sec = wires_df['sector'].max()

# min_st = wires_df['station'].min()
# max_st = wires_df['station'].max()


# def compute_phi_acceptance(acc=True, rang=False):
#     acceptances = np.full((max_wh * 2 + 1, max_sec, max_st, 2), None, dtype=object)
#     ranges = np.full((max_wh * 2 + 1, max_sec, max_st, 2), None, dtype=object)
#     # phi1 = []
#     # phi2 = []

#     for wh in range(min_wh, max_wh + 1):
#         for sec in range(min_sec, max_sec + 1):
#             for st in range(min_st, max_st + 1):
#                 # print(f'Wheel: {wh}, Sector: {sec}, Station: {st}')
#                 if acc:
#                     phi1_st_acc, phi2_st_acc =  chamber_phi_acceptance(wh, sec, st)
#                     acceptances[wh + 2, sec - 1, st - 1] = [phi1_st_acc, phi2_st_acc]
#                 if rang:
#                     phi1_st_rang, phi2_st_rang =  chamber_phi_range(wh, sec, st)
#                     ranges[wh + 2, sec - 1, st - 1] = [phi1_st_rang, phi2_st_rang]
#                 # print(f'Phi1: {phi1_st}, Phi2: {phi2_st}')
#                 # phi1.append(phi1_st)
#                 # phi2.append(phi2_st)
#     return ranges, acceptances

# rang = False

# ranges, acceptances = compute_phi_acceptance(acc=True, rang=rang)

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# wh = 0
# if rang:
#     for st in range(1 + min_st, max_st + 1):
#         phi1 = ranges[wh + 2, :, st - 1, 0]
#         phi2 = ranges[wh + 2, :, st - 1, 1]

#         phi1 = np.array([p for p in phi1 if p is not None], dtype=float)
#         phi2 = np.array([p for p in phi2 if p is not None], dtype=float)
#         x_phi1 = st * np.cos(phi1)
#         y_phi1 = st * np.sin(phi1)
#         x_phi2 = st * np.cos(phi2)
#         y_phi2 = st * np.sin(phi2)

#         start_points = np.array([[x_phi1, y_phi1]])
#         end_points = np.array([[x_phi2, y_phi2]])

#         for start, end in zip(start_points, end_points):
#             x_values = [start[0], end[0]]
#             y_values = [start[1], end[1]]
#             ax.plot(x_values, y_values, c='k', linestyle='-')

##########################################

# for st in range(min_st, max_st + 1):
#     for sec in range(min_sec, max_sec + 1):
#         phi1 = acceptances[wh + 2, sec - 1, st - 1, 0]
#         phi2 = acceptances[wh + 2, sec - 1, st - 1, 1]

#         if phi1 == None or phi2 == None: continue
#         x_phi1 = (cpa.wires_df[(cpa.wires_df['wheel'] == wh) & (cpa.wires_df['sector'] == sec) & (cpa.wires_df['station'] == st) & (cpa.wires_df['phi'] == phi1)]['global_x']).values
#         y_phi1 = (cpa.wires_df[(cpa.wires_df['wheel'] == wh) & (cpa.wires_df['sector'] == sec) & (cpa.wires_df['station'] == st) & (cpa.wires_df['phi'] == phi1)]['global_y']).values
#         x_phi2 = (cpa.wires_df[(cpa.wires_df['wheel'] == wh) & (cpa.wires_df['sector'] == sec) & (cpa.wires_df['station'] == st) & (cpa.wires_df['phi'] == phi2)]['global_x']).values
#         y_phi2 = (cpa.wires_df[(cpa.wires_df['wheel'] == wh) & (cpa.wires_df['sector'] == sec) & (cpa.wires_df['station'] == st) & (cpa.wires_df['phi'] == phi2)]['global_y']).values
#         ax.plot([x_phi1[0], x_phi2[0]], [y_phi1[0], y_phi2[0]], c='limegreen', linestyle='-')

# max_global_x = cpa.wires_df['global_x'].max()
# max_global_y = cpa.wires_df['global_y'].max()
# max_r_global = (max_global_x + max_global_y)**0.5
# for sec in range(1, cpa.max_sec + 1):
#     phi1_sec = acceptances[0, sec - 1, cpa.max_st - 1, 0]
#     phi2_sec = acceptances[0, sec - 1, cpa.max_st - 1, 1]
#     if phi1_sec == None or phi2_sec == None: continue
#     if sec == 7:
#         phi_text = (phi1_sec + phi2_sec) / 2 + np.pi
#     else:
#         phi_text = (phi1_sec + phi2_sec) / 2
#     if phi1_sec < 0 and phi2_sec < 0: x_text, y_text = (max_r_global + 15) * np.cos(phi_text), (max_r_global + 15) * np.sin(phi_text)
#     else: x_text, y_text = (max_r_global) * np.cos(phi_text), (max_r_global) * np.sin(phi_text)

#     ax.text(x_text, y_text, str(sec), fontsize=10)

##########################################

# for st in range(min_st, max_st + 1):
#     phi1 = acceptances[wh + 2, :, st - 1, 0]
#     phi2 = acceptances[wh + 2, :, st - 1, 1]
#     phi1 = np.array([p for p in phi1 if p is not None], dtype=float)
#     phi2 = np.array([p for p in phi2 if p is not None], dtype=float)
#     x_phi1 = st * np.cos(phi1)
#     y_phi1 = st * np.sin(phi1)
#     x_phi2 = st * np.cos(phi2)
#     y_phi2 = st * np.sin(phi2)

#     start_points = np.array([[x_phi1, y_phi1]])
#     end_points = np.array([[x_phi2, y_phi2]])

#     for start, end in zip(start_points, end_points):
#         x_values = [start[0], end[0]]
#         y_values = [start[1], end[1]]
#         ax.plot(x_values, y_values, c='limegreen', linestyle='-')
    # label1 = None
    # label2 = None
    # if st == 1 + min_st:
    #     label1 = 'Phi1'
    #     label2 = 'Phi2'
    # ax.scatter(start_points[:, 0], start_points[:, 1], color='blue', label=label1)
    # ax.scatter(end_points[:, 0], end_points[:, 1], color='red', label=label2)

    # for start, end in zip(start_points, end_points):
    #     print(start)
    #     x_values = [start[0], end[0]]
    #     y_values = [start[1], end[1]]
    #     ax.plot(x_values, y_values, c='limegreen', linestyle='-')

# for sec in range(1, cpa.max_sec + 1):
#     phi1_sec = acceptances[0, sec - 1, cpa.max_st - 1, 0]
#     phi2_sec = acceptances[0, sec - 1, cpa.max_st - 1, 1]
#     if phi1_sec == None or phi2_sec == None: continue
#     if sec == 7:
#         phi_text = (phi1_sec + phi2_sec) / 2 + np.pi
#     else:
#         phi_text = (phi1_sec + phi2_sec) / 2
#     if phi1_sec < 0 and phi2_sec < 0: x_text, y_text = (cpa.max_st + 0.2) * np.cos(phi_text), (cpa.max_st + 0.2) * np.sin(phi_text)
#     else: x_text, y_text = (cpa.max_st + 0.1) * np.cos(phi_text), (cpa.max_st + 0.1) * np.sin(phi_text)

#     ax.text(x_text, y_text, str(sec), fontsize=10)

# if rang:
#     ax.lines[0].set_label('Chambers_max_range')
# ax.lines[-1].set_label('Chambers_acceptance')
# ax.set_title(r"Chamber's acceptance by wires' $\phi$ position")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.legend()
# ax.grid()
# fig.savefig('chambers_acceptance.png')
# fig.savefig('chambers_acceptance.pdf')

# plt.plot(x_phi1[neg1], y_phi1[neg1], 'r+', label='Phi1')
# plt.plot(x_phi1[~neg1], y_phi1[~neg1], 'b+', label='Phi1')
# plt.plot(x_phi2[neg2], y_phi2[neg2], 'r*', label='Phi2')
# plt.plot(x_phi2[~neg2], y_phi2[~neg2], 'b*', label='Phi2')

# plt.plot(x_phi1, y_phi1, 'r+', label='Phi1', markersize=2)
# plt.plot(x_phi2, y_phi2, 'b*', label='Phi2')

# plt.scatter(x_phi1, y_phi1, c='r', label='Phi1')
# plt.scatter(x_phi2, y_phi2, c='b', label='Phi2')
