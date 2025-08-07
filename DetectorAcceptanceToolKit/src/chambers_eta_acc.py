import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import zip_longest

eta_methods = ["SL2_0", "SL2_L1", "SL1_0", "SL1_L2"]

class Ch_eta_acc:

    def __init__(self, verbosity=False):
        self.wires_df = pd.read_csv('files/data/wires_LUT.txt', delim_whitespace=True)
        self.min_wh = self.wires_df['wheel'].min()
        self.max_wh = self.wires_df['wheel'].max()

        self.min_sec = self.wires_df['sector'].min()
        self.max_sec = self.wires_df['sector'].max()

        self.min_st = self.wires_df['station'].min()
        self.max_st = self.wires_df['station'].max()

        self.verbosity = verbosity

        self.acceptances = None

        self.ith = 3  # Default value for ith, can be changed later

    def _get_r_and_z(self, wire_df, eta):
        global_z = wire_df['global_z'].values[0]
        global_x = wire_df['global_x'].values[0]
        global_y = wire_df['global_y'].values[0]
        gloabl_r = (global_x**2 + global_y**2)**0.5
        shift_z = wire_df['length'].values[0] / 2 - 15
        if eta == 1:
            global_z_shifted = global_z - shift_z
        elif eta == 2:
            global_z_shifted = global_z + shift_z
        else:
            raise ValueError("eta must be either 1 or 2.")
        return gloabl_r, global_z_shifted


    def _get_layer_eta1_eta2(self, layer_df):
        layer_eta1_wire_df = layer_df[layer_df['wire'] == round(layer_df['wire'].mean())]
        layer_eta2_wire_df = layer_df[layer_df['wire'] == round(layer_df['wire'].mean())]
        global_r_eta1, shifted_z_eta1 = self._get_r_and_z(layer_eta1_wire_df, 1)
        global_r_eta2, shifted_z_eta2 = self._get_r_and_z(layer_eta2_wire_df, 2)

        eta1, eta2 = -1 * np.log(np.tan(np.arctan2(global_r_eta1, shifted_z_eta1) / 2)), -1 * np.log(np.tan(np.arctan2(global_r_eta2, shifted_z_eta2) / 2))
        return eta1, eta2

    def _chamber_eta_acceptance_SL2_0(self, wh, sec, st):
        SuperLayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == 2)]
        if SuperLayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None

        num_layers = int(SuperLayer_df['layer'].max())
        min_diff = 1000
        eta1 = -5
        eta2 = 5
        for i in range(1, num_layers + 1):
            i_layer_df = SuperLayer_df[SuperLayer_df['layer'] == i]
            min_eta_layer = i_layer_df['eta'].min()
            for j in range(1, num_layers + 1):
                j_layer_df = SuperLayer_df[SuperLayer_df['layer'] == j]
                max_eta_layer = j_layer_df['eta'].max()
                diff = abs(max_eta_layer - min_eta_layer)
                if diff < min_diff:
                    min_diff = diff
                    eta1 = min_eta_layer
                    eta2 = max_eta_layer
        return eta1, eta2
    
    def _chamber_eta_acceptance_SL2_L1(self, wh, sec, st):
        SuperLayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == 2)]
        if SuperLayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: 2")
            return None, None

        i = 1
        L1_df = SuperLayer_df[SuperLayer_df['layer'] == i]
        # return i_layer_df['eta'].min(), i_layer_df['eta'].max()
        sorted_layer = L1_df.sort_values(by='eta')['eta']
        eta1 = sorted_layer.iloc[self.ith]
        eta2 = sorted_layer.iloc[-self.ith - 1]
        return eta1, eta2

    def _chamber_eta_acceptance_SL1_0(self, wh, sec, st, sl=1):
        SuperLayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == sl)]
        if SuperLayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None
        num_layers = int(SuperLayer_df['layer'].max())
        min_diff = 1000
        eta1 = -5
        eta2 = 5
        for i in range(1, num_layers + 1):
            i_layer_df = SuperLayer_df[SuperLayer_df['layer'] == i]
            i_eta1_layer = self._get_layer_eta1_eta2(i_layer_df)[0]

            for j in range(1, num_layers + 1):
                j_layer_df = SuperLayer_df[SuperLayer_df['layer'] == j]
                j_eta2_layer = self._get_layer_eta1_eta2(j_layer_df)[1]
                diff = abs(j_eta2_layer - i_eta1_layer)
                if diff < min_diff:
                    min_diff = diff
                    eta1 = i_eta1_layer
                    eta2 = j_eta2_layer
        return eta1, eta2

    def _chamber_eta_acceptance_SL1_L2(self, wh, sec, st, sl=1):
        SuperLayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == sl)]
        if SuperLayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None
        num_layers = int(SuperLayer_df['layer'].max())
        min_diff = 1000
        layer = 2
        layer_df = SuperLayer_df[SuperLayer_df['layer'] == layer]
        eta1, eta2 = self._get_layer_eta1_eta2(layer_df)
        return eta1, eta2
    
    # def _chamber_eta_acceptance_SL1_L1(self, wh, sec, st):
    #     SuperLayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == 1)]
    #     if SuperLayer_df.empty:
    #         if (self.verbosity):
    #             print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: 1")
    #         return None, None

    #     i = 1
    #     L1_df = SuperLayer_df[SuperLayer_df['layer'] == i]
    #     # return i_layer_df['eta'].min(), i_layer_df['eta'].max()
    #     ith = 1
    #     sorted_layer = L1_df.sort_values(by='eta')['eta']
    #     eta1 = sorted_layer.iloc[ith]
    #     eta2 = sorted_layer.iloc[-ith - 1]
    #     return eta1, eta2

    def compute_eta_acceptance(self, method="SL2_1"):
        acceptances = np.full((self.max_wh * 2 + 1, self.max_sec, self.max_st, 2), None, dtype=object)

        if method not in eta_methods:
            raise ValueError("Choose a method to compute eta within these four: " + str(eta_methods))

        for wh in range(self.min_wh, self.max_wh + 1):
            for sec in range(self.min_sec, self.max_sec + 1):
                for st in range(self.min_st, self.max_st + 1):
                    if (self.verbosity):
                        print(f"Computing eta acceptance for Wheel: {wh}, Sector: {sec}, Station: {st}")
                    if method == eta_methods[0]:
                        eta1_st_acc, eta2_st_acc =  self._chamber_eta_acceptance_SL2_0(wh, sec, st)
                    elif method == eta_methods[1]:
                        eta1_st_acc, eta2_st_acc =  self._chamber_eta_acceptance_SL2_L1(wh, sec, st)
                    elif method == eta_methods[2]:
                        eta1_st_acc, eta2_st_acc =  self._chamber_eta_acceptance_SL1_0(wh, sec, st)
                    elif method == eta_methods[3]:
                        eta1_st_acc, eta2_st_acc =  self._chamber_eta_acceptance_SL1_L2(wh, sec, st)
                    acceptances[wh + 2, sec - 1, st - 1] = [eta1_st_acc, eta2_st_acc]
        self.acceptances = acceptances
        return acceptances
    
    def save_acceptances_to_txt(self, sec=1):
        # This function creates a .txt file in csv format (delimiters are whitespaces) given a sector where eta acceptances for every station and wheel are saved.
        if not np.all(self.acceptances == None):
            print("Saving acceptances to files/output/eta_acceptances.txt")
            eta_MB = self.acceptances[:, sec - 1, :, :]
            eta1_MB1 = eta_MB[:, 0, 0]
            eta2_MB1 = eta_MB[:, 0, 1]
            eta1_MB2 = eta_MB[:, 1, 0]
            eta2_MB2 = eta_MB[:, 1, 1]
            eta1_MB3 = eta_MB[:, 2, 0]
            eta2_MB3 = eta_MB[:, 2, 1]
            eta1_MB4 = eta_MB[:, 3, 0]
            eta2_MB4 = eta_MB[:, 3, 1]
            acceptances_df = pd.DataFrame(list(zip_longest(eta1_MB1, eta1_MB2, eta1_MB3, eta1_MB4, eta2_MB1, eta2_MB2, eta2_MB3, eta2_MB4, fillvalue=np.nan)), columns=['eta1_MB1', 'eta1_MB2', 'eta1_MB3', 'eta1_MB4', 'eta2_MB1', 'eta2_MB2', 'eta2_MB3', 'eta2_MB4'])
            acceptances_df.to_csv('files/output/eta_acceptances.txt', sep=' ', index=False, na_rep='NaN')
            return 1
        return -1
    
    def save_acceptances_as_np_obj(self):
        print("Saving acceptances to files/output/eta_acceptances.npy")
        np.save("files/output/eta_acceptances.npy", self.acceptances)
