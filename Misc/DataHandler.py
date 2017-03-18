import platform
import os
import glob

import numpy as np
from matplotlib import pyplot


class Data_Handler():
    def __init__(self, X, T):
        if (platform.system() == 'Windows'):
            self.fPath = 'dataFiles\\'
            self.iPath = 'images\\'
        else:
            self.fPath = 'dataFiles/'
            self.iPath = 'images/'
        if not os.path.exists("dataFiles"):
            os.makedirs("dataFiles")
        if not os.path.exists("images"):
            os.makedirs("images")
        filelist = glob.glob(self.iPath + "*.png")
        for f in filelist:
            os.remove(f)
        self.T = T
        self.X = X
        self.X_Mesh, self.T_Mesh = np.meshgrid(X, T)

    def save_data(self, data, fileName):
        np.save(self.fPath + fileName + ".npy", data)
        return

    def plot_data(self, data, figName, xLabel, yLabel, scale=None, extra=None):
        fig = pyplot.figure(figName)
        pyplot.title(figName)
        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if scale is not None:
            plt = pyplot.contourf(self.X_Mesh, self.T_Mesh, data, levels=scale, extend='both', cmap=pyplot.get_cmap("inferno"))
            ticks = [scale[0], scale[int(len(scale)/4)], scale[int(len(scale)/2)], scale[int(len(scale)*3/4)], scale[-1]]
        else:
            plt = pyplot.contourf(self.X_Mesh, self.T_Mesh, data, extend='both')
            ticks = None
        fig.colorbar(plt, ticks=ticks)
        pyplot.savefig(self.iPath + figName + ".png")

    def plot_atom(self, data, figName, xLabel, yLabel, dataLabel):
        pyplot.figure(figName)
        pyplot.title(figName)
        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        pyplot.plot(self.T, data, label = dataLabel)
        pyplot.legend()
        pyplot.savefig(self.iPath + figName + ".png")

    def plot_local_field(self, data, figName, xLabel, yLabel, dataLabel):
        pyplot.figure(figName)
        pyplot.title(figName)
        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        pyplot.ylabel(yLabel)
        pyplot.plot(self.T, data, label = dataLabel)
        pyplot.legend()
        pyplot.savefig(self.iPath + figName + ".png")
