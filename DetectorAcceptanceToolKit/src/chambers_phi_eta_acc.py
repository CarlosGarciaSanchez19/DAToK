import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import mplhep as hep
from src.chambers_eta_acc import Ch_eta_acc
from src.chambers_phi_acc import Ch_phi_acc

hep.style.use("CMS")

class Ch_phi_eta_acc:
    def __init__(self, verbosity=False, eta_acc_file="", phi_acc_file=""):

        if eta_acc_file != "" or phi_acc_file != "":
            if not eta_acc_file.endswith(".npy") or not phi_acc_file.endswith(".npy"):
                raise ValueError("eta_acc_file and phi_acc_file must have the right format (.npy)")

        self.verbosity = verbosity

        self.cea = Ch_eta_acc(verbosity=verbosity)
        self.cpa = Ch_phi_acc(verbosity=verbosity)

        self.min_wh = self.cpa.min_wh
        self.max_wh = self.cpa.max_wh

        self.min_sec = self.cpa.min_sec
        self.max_sec = self.cpa.max_sec

        self.min_st = self.cpa.min_st
        self.max_st = self.cpa.max_st

        if eta_acc_file == "":
            self.eta_acceptances = self.cea.compute_eta_acceptance()
        else:
            self.eta_acceptances = np.load("files/output/" + eta_acc_file, allow_pickle=True)
        if phi_acc_file == "":
            self.phi_acceptances = self.cpa.compute_phi_acceptance()[0]
        else:
            self.phi_acceptances = np.load("files/output/" + phi_acc_file, allow_pickle=True)
    
    def save_eta_acceptances_to_txt(self, sec=1):
        self.cea.save_acceptances_to_txt(sec=sec)

    def save_phi_acceptances_to_txt(self, wh=0):
        self.cpa.save_acceptances_to_txt(wh=wh)
    
    def save_eta_acceptances_as_np_obj(self):
        self.cea.save_acceptances_as_np_obj()
    
    def save_phi_acceptances_as_np_obj(self):
        self.cpa.save_acceptances_as_np_obj()
    
    def plot2D_map(self, st=1):
        fig, ax = plt.subplots()
        for wh in range(self.min_wh, self.max_wh + 1):
            for sec in range(self.min_sec, self.max_sec + 1):
                eta1 = self.eta_acceptances[wh + 2, sec - 1, st - 1, 0]
                eta2 = self.eta_acceptances[wh + 2, sec - 1, st - 1, 1]
                phi1 = self.phi_acceptances[wh + 2, sec - 1, st - 1, 0]
                phi2 = self.phi_acceptances[wh + 2, sec - 1, st - 1, 1]
                # if self.verbosity and st == 4:
                #     print(f"Plotting acceptance for wheel {wh}, sector {sec} and station MB{st}:")
                #     print(f"Eta range:")
                #     print(eta1, eta2)
                #     print(f"Phi range:")
                #     print(phi1, phi2)
                if eta1 == None or eta2 == None or phi1 == None or phi2 == None: continue
                
                if eta1 < -1.2:
                    eta1 = -1.2
                if eta2 > 1.2:
                    eta2 = 1.2
                if sec == 7:
                    ax.fill_between([phi1, 3.2], y1=eta1, y2=eta2, color="limegreen")
                    ax.fill_between([-3.2, phi2], y1=eta1, y2=eta2, color="limegreen")
                else:
                    ax.fill_between([phi1, phi2], y1=eta1, y2=eta2, color="limegreen")
        # xmin, xmax = [-3.2, 3.2]
        # ymin, ymax = [-1.4, 1.4]
        fontsize = 20
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-1.4, 1.4)
        ax.text(
            0.01, 1.05,
            "Private work (CMS LUT information)",
            fontsize=fontsize,
            verticalalignment='top',
            fontproperties="Tex Gyre Heros:italic",
            transform=ax.transAxes
        )
        ax.text(
            0.9, 1.05,
            "MB" + str(st),
            fontsize=fontsize,
            verticalalignment='top',
            fontproperties="Tex Gyre Heros:bold",
            transform=ax.transAxes
        )
        ax.set_xlabel(r"global $\phi$", fontweight='bold', fontsize=fontsize)
        ax.set_ylabel(r"global $\eta$", fontweight='bold', fontsize=fontsize)
        ax.grid(True, which='major', linestyle=':', linewidth=0.4, color='k')
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='k')
        print("Saving map as eta_phi_map_MB" + str(st) + ".png")
        fig.savefig("eta_phi_map_MB" + str(st) + ".png")
        print("Saving map as eta_phi_map_MB" + str(st) + ".pdf")
        fig.savefig("eta_phi_map_MB" + str(st) + ".pdf")
