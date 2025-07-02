import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import zip_longest

class ch_phi_acc:

    def __init__(self, verbosity=False):
        self.wires_df = pd.read_csv('wires_LUT.txt', delim_whitespace=True)
        self.min_wh = self.wires_df['wheel'].min()
        self.max_wh = self.wires_df['wheel'].max()

        self.min_sec = self.wires_df['sector'].min()
        self.max_sec = self.wires_df['sector'].max()

        self.min_st = self.wires_df['station'].min()
        self.max_st = self.wires_df['station'].max()

        self.acceptances = None
        self.ranges = None

        self.verbosity = verbosity

    def _chamber_phi_range(self, wh, sec, st, sl=1):
        Slayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == sl)]
        if Slayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None

        num_layers = int(Slayer_df['layer'].max())
        pos_phi = Slayer_df[Slayer_df['phi'] > 0]['phi']
        neg_phi = Slayer_df[Slayer_df['phi'] < 0]['phi']
        if sec == 7:
            phi1, phi2 = pos_phi.min(), neg_phi.max()
        else:
            phi1, phi2 = Slayer_df['phi'].min(), Slayer_df['phi'].max()
        return phi1, phi2
    
    def _chamber_phi_acceptance_0(self, wh, sec, st, sl=1):
        Slayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == sl)]
        if Slayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None

        num_layers = int(Slayer_df['layer'].max())
        min_diff = 1000
        phi1 = -5
        phi2 = 5
        for i in range(1, num_layers + 1):
            i_layer_df = Slayer_df[Slayer_df['layer'] == i]
            if sec == 7:
                min_phi_layer = i_layer_df[i_layer_df['phi'] > 0]['phi'].min()
            else:
                min_phi_layer = i_layer_df['phi'].min()

            for j in range(1, num_layers + 1):
                j_layer_df = Slayer_df[Slayer_df['layer'] == j]
                if sec == 7:
                    max_phi_layer = j_layer_df[j_layer_df['phi'] < 0]['phi'].max()
                    diff = abs(abs(max_phi_layer) - min_phi_layer)
                else:
                    max_phi_layer = j_layer_df['phi'].max()
                    diff = abs(max_phi_layer - min_phi_layer)
                if diff < min_diff:
                    min_diff = diff
                    phi1 = min_phi_layer
                    phi2 = max_phi_layer
        return phi1, phi2

    def _chamber_phi_acceptance_1(self, wh, sec, st, sl=1):
        Slayer_df = self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['super_layer'] == sl)]
        if Slayer_df.empty:
            if (self.verbosity):
                print(f"No data found for Wheel: {wh}, Sector: {sec}, Station: {st}, Super Layer: {sl}")
            return None, None

        i = 1
        j = 1
        i_layer_df = Slayer_df[Slayer_df['layer'] == i]
        j_layer_df = Slayer_df[Slayer_df['layer'] == j]
        if sec == 7:
            min_phi_layer, max_phi_layer = i_layer_df[i_layer_df['phi'] > 0]['phi'].min(), j_layer_df[j_layer_df['phi'] < 0]['phi'].max()
        else:
            min_phi_layer, max_phi_layer = i_layer_df['phi'].min(), j_layer_df['phi'].max()
        phi1 = min_phi_layer
        phi2 = max_phi_layer
        return phi1, phi2

    def compute_phi_acceptance(self, acc=True, rang=False, kind=0):
        if not acc and not rang:
            print("Computing nothing...")
            print("Try to set one of the arguments (acc or rang) to True.")
            return np.array((None)), np.array((None))
        acceptances = np.full((self.max_wh * 2 + 1, self.max_sec, self.max_st, 2), None, dtype=object)
        ranges = np.full((self.max_wh * 2 + 1, self.max_sec, self.max_st, 2), None, dtype=object)

        for wh in range(self.min_wh, self.max_wh + 1):
            for sec in range(self.min_sec, self.max_sec + 1):
                for st in range(self.min_st, self.max_st + 1):
                    if (self.verbosity):
                        print(f"Computing phi acceptance for Wheel: {wh}, Sector: {sec}, Station: {st}")
                    if acc:
                        if kind == 0:
                            phi1_st_acc, phi2_st_acc =  self._chamber_phi_acceptance_0(wh, sec, st)
                        elif kind == 1:
                            phi1_st_acc, phi2_st_acc =  self._chamber_phi_acceptance_1(wh, sec, st)
                        acceptances[wh + 2, sec - 1, st - 1] = [phi1_st_acc, phi2_st_acc]
                    if rang:
                        phi1_st_rang, phi2_st_rang =  self._chamber_phi_range(wh, sec, st)
                        ranges[wh + 2, sec - 1, st - 1] = [phi1_st_rang, phi2_st_rang]
        if acc:
            self.acceptances = acceptances
        if rang:
            self.ranges = ranges
        return ranges, acceptances
    
    def _to_valid_values(self, vec):
        vec_valid = np.array([x for x in vec if x is not None], dtype=float)
        return vec_valid

    def save_acceptances(self):
        if not np.all(self.acceptances == None):
            print("Saving acceptances to phi_acceptances.txt")
            time.sleep(0.5)
            phi_MB = self.acceptances[0, :, :, :]
            phi1_MB1 = phi_MB[:, 0, 0]
            phi2_MB1 = phi_MB[:, 0, 1]
            phi1_MB2 = phi_MB[:, 1, 0]
            phi2_MB2 = phi_MB[:, 1, 1]
            phi1_MB3 = phi_MB[:, 2, 0]
            phi2_MB3 = phi_MB[:, 2, 1]
            phi1_MB4 = phi_MB[:, 3, 0]
            phi2_MB4 = phi_MB[:, 3, 1]
            # phi_MB2 = phi_MB[:, 1, :]
            # phi_MB3 = phi_MB[:, 2, :]
            # phi_MB4 = phi_MB[:, 3, :]
            # phi_MB1 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB1[:, 0], phi_MB1[:, 1]]))))
            # phi_MB2 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB2[:, 0], phi_MB2[:, 1]]))))
            # phi_MB3 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB3[:, 0], phi_MB3[:, 1]]))))
            # phi_MB4 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB4[:, 0], phi_MB4[:, 1]]))))
            # print(type(phi_MB1), phi_MB1)
            acceptances_df = pd.DataFrame(list(zip_longest(phi1_MB1, phi1_MB2, phi1_MB3, phi1_MB4, phi2_MB1, phi2_MB2, phi2_MB3, phi2_MB4, fillvalue=np.nan)), columns=['phi1_MB1', 'phi1_MB2', 'phi1_MB3', 'phi1_MB4', 'phi2_MB1', 'phi2_MB2', 'phi2_MB3', 'phi2_MB4'])
            acceptances_df.to_csv('phi_acceptances.txt', sep=' ', index=False, na_rep='NaN')
        if not np.all(self.ranges == None):
            print("Saving maximum ranges to phi_max_ranges.txt")
            time.sleep(0.5)
            phi_MB = self.ranges[0, :, :, :]
            phi_MB1 = phi_MB[:, 0, :]
            phi_MB2 = phi_MB[:, 1, :]
            phi_MB3 = phi_MB[:, 2, :]
            phi_MB4 = phi_MB[:, 3, :]
            phi_MB1 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB1[:, 0], phi_MB1[:, 1]]))))
            phi_MB2 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB2[:, 0], phi_MB2[:, 1]]))))
            phi_MB3 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB3[:, 0], phi_MB3[:, 1]]))))
            phi_MB4 = np.transpose(np.sort(self._to_valid_values(np.concatenate([phi_MB4[:, 0], phi_MB4[:, 1]]))))
            ranges_df = pd.DataFrame(list(zip_longest(phi_MB1, phi_MB2, phi_MB3, phi_MB4, fillvalue=np.nan)), columns=['MB1', 'MB2', 'MB3', 'MB4'])
            ranges_df.to_csv('phi_max_ranges.txt', sep=' ', index=False, na_rep='NaN')
        if np.all(self.ranges == None) and np.all(self.acceptances == None):
            print("Execute compute_phi_acceptance method before saving.")

    def plot_ch_phi_acc(self, wh=0, acc=True, rang=False, kind=0, outfile=''):

        self.compute_phi_acceptance(acc=acc, rang=rang, kind=kind)
        if np.all(self.ranges == None) and np.all(self.acceptances == None):
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if rang:
            for st in range(self.min_st, self.max_st + 1):
                phi1 = self.ranges[wh + 2, :, st - 1, 0]
                phi2 = self.ranges[wh + 2, :, st - 1, 1]

                phi1 = np.array([p for p in phi1 if p is not None], dtype=float)
                phi2 = np.array([p for p in phi2 if p is not None], dtype=float)
                x_phi1 = st * np.cos(phi1)
                y_phi1 = st * np.sin(phi1)
                x_phi2 = st * np.cos(phi2)
                y_phi2 = st * np.sin(phi2)

                start_points = np.array([[x_phi1, y_phi1]])
                end_points = np.array([[x_phi2, y_phi2]])

                for start, end in zip(start_points, end_points):
                    x_values = [start[0], end[0]]
                    y_values = [start[1], end[1]]
                    ax.plot(x_values, y_values, c='k', linestyle='-')
        if acc:
            for st in range(self.min_st, self.max_st + 1):
                phi1 = self.acceptances[wh + 2, :, st - 1, 0]
                phi2 = self.acceptances[wh + 2, :, st - 1, 1]
                phi1 = np.array([p for p in phi1 if p is not None], dtype=float)
                phi2 = np.array([p for p in phi2 if p is not None], dtype=float)
                x_phi1 = st * np.cos(phi1)
                y_phi1 = st * np.sin(phi1)
                x_phi2 = st * np.cos(phi2)
                y_phi2 = st * np.sin(phi2)

                start_points = np.array([[x_phi1, y_phi1]])
                end_points = np.array([[x_phi2, y_phi2]])

                for start, end in zip(start_points, end_points):
                    x_values = [start[0], end[0]]
                    y_values = [start[1], end[1]]
                    ax.plot(x_values, y_values, c='limegreen', linestyle='-')

        for sec in range(self.min_sec, self.max_sec + 1):
            if acc:
                phi1_sec = self.acceptances[0, sec - 1, self.max_st - 1, 0]
                phi2_sec = self.acceptances[0, sec - 1, self.max_st - 1, 1]
            elif rang:
                phi1_sec = self.ranges[0, sec - 1, self.max_st - 1, 0]
                phi2_sec = self.ranges[0, sec - 1, self.max_st - 1, 1]
            if phi1_sec == None or phi2_sec == None: continue
            if sec == 7:
                phi_text = (phi1_sec + phi2_sec) / 2 + np.pi
            else:
                phi_text = (phi1_sec + phi2_sec) / 2
            if phi1_sec < 0 and phi2_sec < 0: x_text, y_text = (self.max_st + 0.2) * np.cos(phi_text), (self.max_st + 0.2) * np.sin(phi_text)
            else: x_text, y_text = (self.max_st + 0.1) * np.cos(phi_text), (self.max_st + 0.1) * np.sin(phi_text)

            ax.text(x_text, y_text, str(sec), fontsize=10)
        ax.set_title(r"Chamber's acceptance by wires' $\phi$ position")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
        if rang and acc:
            ax.lines[0].set_label('Chambers_max_range')
            ax.lines[-1].set_label('Chambers_acceptance')
            if outfile == '':
                outfile = 'chambers_acceptance_and_max_range'
        elif acc:
            ax.lines[-1].set_label('Chambers_acceptance')
            if outfile == '':
                outfile = 'chambers_acceptance'
        elif rang:
            ax.lines[-1].set_label('Chambers_max_range')
            if outfile == '':
                outfile = 'chambers_max_range'

        ax.grid()
        ax.legend()
        print("Plotting acceptances...")
        time.sleep(1)
        print("Saving plot as "+ outfile +".png")
        fig.savefig(outfile + ".png")
        print("Saving plot as "+ outfile +".pdf")
        fig.savefig(outfile + ".pdf")
        return
        
    def plot_ch_phi_acc_wire_xy_pos(self, wh=0):

        self.compute_phi_acceptance(True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for st in range(self.min_st, self.max_st + 1):
            for sec in range(self.min_sec, self.max_sec + 1):
                phi1 = self.acceptances[wh + 2, sec - 1, st - 1, 0]
                phi2 = self.acceptances[wh + 2, sec - 1, st - 1, 1]

                if phi1 == None or phi2 == None: continue
                x_phi1 = (self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['phi'] == phi1)]['global_x']).values
                y_phi1 = (self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['phi'] == phi1)]['global_y']).values
                x_phi2 = (self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['phi'] == phi2)]['global_x']).values
                y_phi2 = (self.wires_df[(self.wires_df['wheel'] == wh) & (self.wires_df['sector'] == sec) & (self.wires_df['station'] == st) & (self.wires_df['phi'] == phi2)]['global_y']).values
                ax.plot([x_phi1[0], x_phi2[0]], [y_phi1[0], y_phi2[0]], c='limegreen', linestyle='-')

        max_global_x = self.wires_df['global_x'].max()
        max_global_y = self.wires_df['global_y'].max()
        max_r_global = ((max_global_x)**2 + (max_global_y)**2)**0.5*0.7
        for sec in range(self.min_sec, self.max_sec + 1):
            phi1_sec = self.acceptances[wh + 2, sec - 1, self.max_st - 1, 0]
            phi2_sec = self.acceptances[wh + 2, sec - 1, self.max_st - 1, 1]
            if phi1_sec == None or phi2_sec == None: continue
            if sec == 7:
                phi_text = (phi1_sec + phi2_sec) / 2 + np.pi
            else:
                phi_text = (phi1_sec + phi2_sec) / 2
            if phi1_sec < 0 and phi2_sec < 0: x_text, y_text = (max_r_global) * np.cos(phi_text), (max_r_global) * np.sin(phi_text)
            else: x_text, y_text = (max_r_global) * np.cos(phi_text), (max_r_global) * np.sin(phi_text)

            ax.text(x_text, y_text, str(sec), fontsize=10)

        ax.lines[-1].set_label('Chambers_acceptance')
        ax.set_title(r"Chamber's acceptance by wires' $\phi$ position (real wire pos)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.grid()

        print("Plotting acceptances...")
        time.sleep(1)
        print("Saving plot as chambers_acceptance_wire_xy_pos.png")
        fig.savefig("chambers_acceptance_wire_xy_pos.png")
        print("Saving plot as chambers_acceptance_wire_xy_pos.pdf")
        fig.savefig("chambers_acceptance_wire_xy_pos.pdf")

