import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from src.chambers_eta_acc import Ch_eta_acc
from src.chambers_phi_acc import Ch_phi_acc

class Ch_phi_eta_acc:
    def __init__(self, verbosity=False):
        self.cea = Ch_eta_acc(verbosity=verbosity)
        self.cpa = Ch_phi_acc(verbosity=verbosity)

        self.min_wh = self.cpa.min_wh
        self.max_wh = self.cpa.max_wh

        self.min_sec = self.cpa.min_sec
        self.max_sec = self.cpa.max_sec

        self.min_st = self.cpa.min_st
        self.max_st = self.cpa.max_st

        self.eta_acceptances = self.cea.acceptances
        self.phi_acceptances = self.cpa.acceptances
    
    def save_phi_acceptances_to_txt(self, wh=0):
        self.cpa.save_acceptances_to_txt(wh=wh)
    
    def save_eta_acceptances_to_txt(self, sec=1):
        self.cea.save_acceptances_to_txt(sec=sec)
    
    def plot2D_map(self, st=1):
        print("Saving map as eta_phi_map_MB" + str(st) + ".png")
        print("Saving map as eta_phi_map_MB" + str(st) + ".pdf")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for wh in range(self.min_wh, self.max_wh + 1):
            for sec in range(self.min_sec, self.max_sec + 1):
                y1 = self.eta_acceptances[wh + 2, sec - 1, st - 1, 0]
                y2 = self.eta_acceptances[wh + 2, sec - 1, st - 1, 1]
                x1 = self.phi_acceptances[wh + 2, sec - 1, st - 1, 0]
                x2 = self.phi_acceptances[wh + 2, sec - 1, st - 1, 1]
                
                if y1 == None or y2 == None or x1 == None or x2 == None: continue
                
                if y1 < -1.2:
                    y1 = -1.2
                if y2 > 1.2:
                    y2 = 1.2
                if sec == 7:
                    ax.fill_between([x1, 3.2], y1=y1, y2=y2, color="limegreen")
                    ax.fill_between([-3.2, x2], y1=y1, y2=y2, color="limegreen")
                else:
                    ax.fill_between([x1, x2], y1=y1, y2=y2, color="limegreen")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid()
        fig.savefig("eta_phi_map_MB" + str(st) + ".png")
        fig.savefig("eta_phi_map_MB" + str(st) + ".pdf")