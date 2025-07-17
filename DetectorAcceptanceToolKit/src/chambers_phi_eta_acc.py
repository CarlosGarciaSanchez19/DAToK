import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import mplhep as hep
from src.chambers_eta_acc import Ch_eta_acc
from src.chambers_phi_acc import Ch_phi_acc

hep.style.use("CMS")

eta_methods = ["SL2_0", "SL2_L1", "SL1_0", "SL1_L2"]
phi_methods = ["SL1_0", "SL1_L1"]

class Ch_phi_eta_acc:

    def __init__(self, verbosity=False, eta_method=eta_methods[1], phi_method=phi_methods[1], eta_acc_file="", phi_acc_file=""):

        if eta_acc_file != "" or phi_acc_file != "":
            if not eta_acc_file.endswith(".npy") or not phi_acc_file.endswith(".npy"):
                raise ValueError("eta_acc_file and phi_acc_file must have the right format (.npy)")
        if eta_method not in ["SL2_0", "SL2_L1", "SL1_0", "SL1_L2", None]:
            raise ValueError("Choose a method to compute eta within these four: " + str(eta_methods))
        if phi_method not in ["SL1_0", "SL1_L1", None]:
            raise ValueError("Choose a method to compute phi within these two: " + str(phi_methods))

        self.verbosity = verbosity
        
        if eta_method == None:
            eta_method = eta_methods[1]
        if phi_method == None:
            phi_method = phi_methods[1]
        print("COMPUTING ACCEPTANCES WITH:")
        print("eta_method: " + eta_method)
        print("phi_method: " + phi_method)
        self.eta_method = eta_method
        self.phi_method = phi_method

        self.cea = Ch_eta_acc(verbosity=verbosity)
        self.cpa = Ch_phi_acc(verbosity=verbosity)

        self.min_wh = self.cpa.min_wh
        self.max_wh = self.cpa.max_wh

        self.min_sec = self.cpa.min_sec
        self.max_sec = self.cpa.max_sec

        self.min_st = self.cpa.min_st
        self.max_st = self.cpa.max_st

        if eta_acc_file == "":
            self.eta_acceptances = self.cea.compute_eta_acceptance(method=eta_method)
        else:
            self.eta_acceptances = np.load("files/output/" + eta_acc_file, allow_pickle=True)
        if phi_acc_file == "":
            self.phi_acceptances = self.cpa.compute_phi_acceptance(method=phi_method)[0]
        else:
            self.phi_acceptances = np.load("files/output/" + phi_acc_file, allow_pickle=True)
        
        if "SL2" in eta_method:
            print("\nNOTE: eta acceptances in MB4 are computed with eta_method='SL1_L2' because there's no SL2 in MB4.\n")
            time.sleep(1)
            min_st = self.cea.min_st
            self.cea.min_st = 4
            self.cea.verbosity = False
            self.eta_acceptances[:, :, 3, :] = self.cea.compute_eta_acceptance(method="SL1_L2")[:, :, 3, :]
            self.cea.verbosity = verbosity
            self.cea.min_st = min_st
        
        if phi_method == "SL1_L1":
            print("\nNOTE: phi acceptances in MB4 are computed using 3rd and 3rd-last wires.\n")
            time.sleep(1)
            min_st = self.cpa.min_st
            self.cpa.min_st = 4
            self.cpa.verbosity = False
            self.phi_acceptances[:, :, 3, :] = self.cpa.compute_phi_acceptance(method="SL1_L1", ith=2)[0][:, :, 3, :]
            self.cpa.verbosity = verbosity
            self.cpa.min_st = min_st
    
    def save_eta_acceptances_to_txt_format(self, sec=1):
        self.cea.save_acceptances_to_txt(sec=sec)

    def save_phi_acceptances_to_txt_format(self, wh=0):
        self.cpa.save_acceptances_to_txt(wh=wh)
    
    def save_eta_acceptances_as_np_obj(self):
        self.cea.save_acceptances_as_np_obj()
    
    def save_phi_acceptances_as_np_obj(self):
        self.cpa.save_acceptances_as_np_obj()
    
    def _write_map(self, h, st, var, eta_phi_map):
        name = "MB" + str(st) + "_" + var
        for wh in range(self.min_wh, self.max_wh + 1):
            max_sec = sum(np.array([x is not None for x in eta_phi_map[wh + 2, :]]))
            for sec in range(self.min_sec, max_sec + 1):
                value = eta_phi_map[wh + 2, sec - 1]
                if wh < 0:
                    wh_label = "Neg" + str(abs(wh))
                else:
                    wh_label = str(wh)
                if wh == self.min_wh and sec == self.min_sec:
                    h.write('std::map<string, float> ' + name + ' = { \n{"wh' + wh_label + '_sec' + str(sec) + '", ' + str(value) + '}')
                elif wh == self.max_wh and sec == max_sec:
                    h.write(', {"wh' + wh_label + '_sec' + str(sec) + '", ' + str(value) + '} \n};\n\n')
                elif sec == max_sec:
                    h.write(', {"wh' + wh_label + '_sec' + str(sec) + '", ' + str(value) + '},\n')
                elif sec == self.min_sec:
                    h.write('{"wh' + wh_label + '_sec' + str(sec) + '", ' + str(value) + '}')
                else:
                    h.write(', {"wh' + wh_label + '_sec' + str(sec) + '", ' + str(value) + '}')

    def save_eta_phi_acceptances_as_Clibrary(self):
        print("Saving acceptances into DTAcceptances.h")
        with open("DTAcceptances.h", 'w') as h:
            h.write("// eta_method: " + self.eta_method)
            if "SL2" in self.eta_method:
                h.write("// eta_method for MB4: " + eta_methods[3])
            h.write("// phi_method: " + self.phi_method)
            h.write("\n#ifndef DTACCEPTANCES_H\n")
            h.write("# define DTACCEPTANCES_H\n\n")
            for st in range(self.min_st, self.max_st + 1):
                phi1 = self.phi_acceptances[:, :, st - 1, 0]
                phi2 = self.phi_acceptances[:, :, st - 1, 1]
                eta1 = self.eta_acceptances[:, :, st - 1, 0]
                eta2 = self.eta_acceptances[:, :, st - 1, 1]
                self._write_map(h, st, "phi1", phi1)
                self._write_map(h, st, "phi2", phi2)
                self._write_map(h, st, "eta1", eta1)
                self._write_map(h, st, "eta2", eta2)
            h.write("#endif\n")
            h.close()

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
            "DT acceptance",
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
