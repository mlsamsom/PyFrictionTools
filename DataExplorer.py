from Tkinter import *
import tkFileDialog
from ttk import Notebook
import os
from filehandling import parse_file
from filehandling import get_filename
from filehandling import decompose_file_name
import cPickle as pickle
from data import FrictionData
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2TkAgg)
import seaborn as sns
from numpy import exp


class DataExploreGUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.nb = Notebook(self)
        self.tab1 = Frame(self.nb)
        self.tab2 = Frame(self.nb)
        self.tab3 = Frame(self.nb)

        self.nb.add(self.tab1, text='Raw Data')
        self.nb.add(self.tab2, text='Quality metrics')
        self.nb.add(self.tab3, text='Friction tests')
        self.nb.grid(row=0, column=0, sticky=NW)

        self.intSettings = {'Group': IntVar(value=0)}
        self.intSettings2 = {'Group': IntVar(value=0)}
        self.intSettings3 = {'Group': IntVar(value=0)}
        self.intSettings4 = {'Group': IntVar(value=0)}

        # Tab1 buttons
        self.buttonLoadFile = Button(self.tab1, text="Load Data File",
                                     command=self.loadFile)
        self.buttonLoadFile.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)
        self.buttonSavePlot = Button(self.tab1, text="Save Plot as SVG",
                                     command=self.savePlotFrame1)
        self.buttonSavePlot.grid(row=0, column=1, padx=5, pady=5,
                                 sticky=W + E)

        self.frameGroups = LabelFrame(self.tab1, text="Group Selection")
        self.frameGroups.grid(row=1, column=0, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.frameGroups, text="").pack(anchor=W)
        self.frameChannels = LabelFrame(self.tab1, text="Channel Selection")
        self.frameChannels.grid(row=1, column=1, padx=5, pady=5,
                                sticky=N + E + W + S)
        Label(self.frameChannels, text="").pack(anchor=W)

        self.buttonPlot = Button(self.tab1, text="Plot Selected Channels",
                                 command=self.plotChannels)
        self.buttonPlot.grid(row=2, column=0, padx=5, pady=5,
                             sticky=W + E)
        self.grid()

        self.fg_sz = (12, 6)
        self.fig = plt.Figure(figsize=self.fg_sz)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab1)
        self.canvas.get_tk_widget().grid(column=2, row=1)

        # Tab2 button
        self.frameGroups3 = LabelFrame(self.tab2, text="Stress Relaxation")
        self.frameGroups3.grid(row=1, column=0, padx=5, pady=5,
                               sticky=N + E + W + S)
        Label(self.frameGroups3, text="").pack(anchor=W)

        self.frameGroups4 = LabelFrame(self.tab2, text="X-Y alignment")
        self.frameGroups4.grid(row=1, column=1, padx=5, pady=5,
                               sticky=N + E + W + S)
        Label(self.frameGroups4, text="").pack(anchor=W)

        self.fig2 = plt.Figure(figsize=self.fg_sz)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().grid(column=2, row=1)

        self.buttonSavePlot2 = Button(self.tab2, text="Save Plot as SVG",
                                      command=self.savePlotFrame2)
        self.buttonSavePlot2.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)

        # Tab3 button
        self.buttonAnalyze = Button(self.tab3, text="Plot friction line",
                                    command=self.plotMu)
        self.buttonAnalyze.grid(row=0, column=0, padx=5, pady=5,
                                sticky=W + E)

        self.frameGroups2 = LabelFrame(self.tab3, text="Friction Run")
        self.frameGroups2.grid(row=1, column=0, padx=5, pady=5,
                               sticky=N + E + W + S)
        Label(self.frameGroups2, text="").pack(anchor=W)

    def loadFile(self):
        # read file into data object
        self.filename = get_filename()
        file_info, repaired = decompose_file_name(self.filename)

        if repaired:
            self.data = cPickle.load(open(self.filename, "rb"))
        else:
            self.data = parse_file(self.filename)

        self.data_obj = FrictionData(self.data, file_info)

        if self.filename:
            for child in self.frameGroups.pack_slaves():
                child.destroy()
            count = 1
            count2 = 1
            self.blockmap = {}
            self.blockmap2 = {}
            for i, g in enumerate(self.data.keys()):
                tag = self.data[g]['tag'][0]
                Radiobutton(self.frameGroups,
                            text='blk' + g[-1] + ': ' + tag[1:-1],
                            indicatoron=0,
                            width=20,
                            variable=self.intSettings["Group"],
                            command=self.populateChannelList,
                            value=i).pack(anchor=W)

                if tag == '<Scripts>':
                    # raw data curves
                    Radiobutton(self.frameGroups2,
                                text='run' + str(count),
                                indicatoron=0,
                                width=20,
                                variable=self.intSettings2["Group"],
                                command=self.plotCurves,
                                value=i).pack(anchor=W)
                    self.blockmap[g] = count - 1
                    # x-y alignment
                    Radiobutton(self.frameGroups4,
                                text='run' + str(count),
                                indicatoron=0,
                                width=20,
                                variable=self.intSettings4["Group"],
                                command=self.plotAlignment,
                                value=i).pack(anchor=W)
                    self.blockmap[g] = count - 1
                    count += 1
                elif tag == '<Stress Relaxation>':
                    Radiobutton(self.frameGroups3,
                                text='Stress Relax' + str(count),
                                indicatoron=0,
                                width=20,
                                variable=self.intSettings3["Group"],
                                command=self.plotProny,
                                value=i).pack(anchor=W)
                    self.blockmap2[g] = count2 - 1
                    count2 += 1

            self.channelSelections = {}
            for c in self.data['block0']['data'].keys():
                if c == 'time':
                    continue
                self.channelSelections[c] = IntVar(value=0)
                Checkbutton(self.frameChannels,
                            text=c,
                            variable=self.channelSelections[c]).pack(anchor=W)

    def populateChannelList(self):
        g = self.data.keys()[self.intSettings['Group'].get()]
        self.channelSelections = {}
        for child in self.frameChannels.pack_slaves():
            child.destroy()
        for c in self.data[g]['data'].keys():
            if c == 'time':
                continue
            self.channelSelections[c] = IntVar(value=0)
            Checkbutton(self.frameChannels,
                        text=c,
                        variable=self.channelSelections[c]).pack(anchor=W)

    def plotChannels(self):
        self.fig = plt.Figure(figsize=self.fg_sz)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab1)
        self.canvas.get_tk_widget().grid(column=2, row=1)

        keys = []
        for c in self.channelSelections.keys():
            if self.channelSelections[c].get():
                keys.append(c)
        block = 'block' + str(self.intSettings["Group"].get())
        for i, k in enumerate(keys):
            self.ax = self.fig.add_subplot(len(keys), 1, i+1, ylabel=k)

            t = self.data[block]['data']['time']
            x = self.data[block]['data'][k][:, 0]
            y = self.data[block]['data'][k][:, 1]
            z = self.data[block]['data'][k][:, 2]

            self.ax.hold(True)
            self.ax.plot(t, x, label='x')
            self.ax.plot(t, y, label='y')
            self.ax.plot(t, z, label='z')
            legend = self.ax.legend(loc='upper right', prop={'size': 3}, shadow=True)

            for label in legend.get_texts():
                label.set_fontsize('small')

        plt.show()

    def savePlotFrame1(self):
        pltsavename = tkFileDialog.asksaveasfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Save As"
        )

        # check input for errors
        pltsavename = ''
        for i in xrange(len(pltsavenamein)):
            if pltsavenamein[i] == '.':
                pltsavename = pltsavename + '_'
            else:
                pltsavename = pltsavename + pltsavenamein[i]

        pltsavename = pltsavename + '.svg'

        if pltsavename:
            self.fig.savefig(pltsavename, format='svg', transparent=True)

    def savePlotFrame2(self):
        pltsavename = tkFileDialog.asksaveasfilename(
            parent=root,
            initialdir=os.getcwd(),
            title="Save As"
        )

        # check input for errors
        pltsavename = ''
        for i in xrange(len(pltsavenamein)):
            if pltsavenamein[i] == '.':
                pltsavename = pltsavename + '_'
            else:
                pltsavename = pltsavename + pltsavenamein[i]

        pltsavename = pltsavename + '.svg'

        if pltsavename:
            self.fig2.savefig(pltsavename, format='svg', transparent=True)

    def plotMu(self):
        self.data_obj.show_friction_line()

    def plotAlignment(self):
        self.fig2 = plt.Figure(figsize=self.fg_sz)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().grid(column=2, row=1)

        g = 'block' + str(self.intSettings4["Group"].get())
        r = float(self.blockmap[g])

        df = self.data_obj.test_summary_frame()

        x = df[df['load_index'] == r]['alig_x'].values
        y = df[df['load_index'] == r]['alig_y'].values

        for i in xrange(2):
            self.ax2 = self.fig2.add_subplot(1, 2, i+1,
                                             ylabel='y position (mm)', xlabel='x position (mm)')
            self.ax2.plot(x[i], y[i], "or")

        plt.show()

    def plotProny(self):
        self.fig2 = plt.Figure(figsize=self.fg_sz)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().grid(column=2, row=1)

        g = 'block' + str(self.intSettings3["Group"].get())
        r = int(self.blockmap2[g])

        df = self.data_obj.test_summary_frame()

        F = self.data_obj.Fz_all[r]

        E = df.loc[r]['E']
        p = df.loc[r]['p']
        tau = df.loc[r]['tau']
        e1 = self.data_obj.e1_all[r]
        t1 = self.data_obj.t1_all[r]
        t = self.data_obj.t_all[r]

        y = prony_hold(t, E, p, tau, e1, t1)

        self.ax3 = self.fig2.add_subplot(1, 1, 1, xlabel='time (s)',
                                         ylabel='Fz (N)')
        self.ax3.hold(True)
        self.ax3.plot(t, F, 'o')
        self.ax3.plot(t, y)

        plt.show()

    def plotCurves(self):
        g = 'block' + str(self.intSettings2["Group"].get())
        r = float(self.blockmap[g])
        self.data_obj.show_friction_curve(run=r)


def prony_hold(t, E, p, tau, e1, t1):
    out = (E * e1 / t1) * (t1 - p * t1 + p * tau * exp(-1 * (t - t1) / tau) -
                           p * tau * exp(-1 * t / tau))
    return out


if __name__ == "__main__":
    root = Tk()
    main = DataExploreGUI(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()
