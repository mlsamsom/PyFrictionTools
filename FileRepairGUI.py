from Tkinter import *
import tkFileDialog
import os
from filehandling import parse_file
import cPickle as pickle
import re


class FileRepairGUI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)

        self._numf = 0
        self._filedata = []
        self._fn = []
        self._address_list = []
        self._block_dict = {}
        self._label_refs = {}
        self._file1_keys = []
        self._file2_keys = []

        self.intSettings = {'Group': IntVar(value=0)}

        self.buttonLoadFile = Button(self, text="Load files to correct", command=self.get_fn)
        self.buttonLoadFile.grid(row=0, column=0, padx=5, pady=5,
                                 sticky=W + E)

        self.button1 = Button(self, text="to file 1", command=self.populate_file1)
        self.button1.grid(row=0, column=1, padx=5, pady=5,
                          sticky=W + E)

        self.button2 = Button(self, text="to file 2", command=self.populate_file2)
        self.button2.grid(row=0, column=2, padx=5, pady=5,
                          sticky=W + E)

        self.savebutton = Button(self, text="Save", command=self.save)
        self.savebutton.grid(row=2, column=self._numf + 1)

        self.cleanFrame1 = LabelFrame(self, text='clean file 1')
        self.cleanFrame1.grid(row=1, column=1, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.cleanFrame1, text="").pack(anchor=W)

        self.cleanFrame2 = LabelFrame(self, text='clean file 2')
        self.cleanFrame2.grid(row=1, column=2, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.cleanFrame2, text="").pack(anchor=W)

        self.Lb1 = None
        self.Lb2 = None

    def get_fn(self):
        """
        Loads a file and puts all blocks in radio buttons.
        :param self:
        :return:
        """
        self.button1.grid_forget()
        self.button2.grid_forget()
        self.cleanFrame1.grid_forget()
        self.cleanFrame2.grid_forget()

        self._numf += 1
        if self._numf > 3:
            raise ValueError("Too many files chosen")

        fn = tkFileDialog.askopenfilename(parent=root, initialdir=os.getcwd(), title="Select a File")
        title = os.path.basename(fn).split('.')[0]

        frameGroups = LabelFrame(self, text=title)
        frameGroups.grid(row=1, column=self._numf - 1, padx=5, pady=5,
                         sticky=N + E + W + S)
        Label(frameGroups, text="").pack(anchor=W)

        self.button1 = Button(self, text="to file 1", command=self.populate_file1)
        self.button1.grid(row=0, column=self._numf, padx=5, pady=5,
                          sticky=W + E)

        self.cleanFrame1 = LabelFrame(self, text='clean file 1')
        self.cleanFrame1.grid(row=1, column=self._numf, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.cleanFrame1, text="").pack(anchor=W)

        self.button2 = Button(self, text="to file 2", command=self.populate_file2)
        self.button2.grid(row=0, column=self._numf + 1, padx=5, pady=5,
                          sticky=W + E)

        self.cleanFrame2 = LabelFrame(self, text='clean file 2')
        self.cleanFrame2.grid(row=1, column=self._numf + 1, padx=5, pady=5,
                              sticky=N + E + W + S)
        Label(self.cleanFrame2, text="").pack(anchor=W)

        dat = parse_file(fn)
        self._filedata.append(dat)
        self._fn.append(fn)

        if fn:
            for child in frameGroups.pack_slaves():
                child.destroy()

        def populate_address_list(text, variable):
            self._address_list.append((text, variable))

        if self._numf == 0:
            raise ValueError("Need to load at least one file")

        for i, f in enumerate(dat.keys()):
            tag = dat[f]['tag'][0]
            info = dat[f]['info']
            if tag == '<Stress Relaxation>':
                lead = 'SR'
                for l in info:
                    if l[0] == 'Amplitude, mm:':
                        amp = l[1]
            elif tag == '<Scripts>':
                lead = 'Fric'
                for l in info:
                    s_obj = re.search(r'1HC\d{1,2}\.\d{4},\d{1,2}\.\d{4}', l[0])
                    if s_obj:
                        amp = s_obj.group()[3:-3]
            else:
                SystemExit("Error block {:d} in file not recognized".format(i))

            button_text = 'b' + f[-1] + ': ' + lead + '-' + amp

            key = f + "_" + str(self._numf)
            self._block_dict[key] = BooleanVar(value=False)
            self._label_refs[key] = button_text

            Checkbutton(frameGroups,
                        text=button_text,
                        variable=self._block_dict[key]).grid(row=i, column=0, sticky=NW)

    def populate_file1(self):
        if self.Lb1 is not None:
            self.Lb1.destroy()

        for c in self._block_dict.keys():
            if self._block_dict[c].get():
                self._file1_keys.append(c)
            self._block_dict[c].set(False)

        self.Lb1 = Listbox(self.cleanFrame1, width=10, selectmode=MULTIPLE)
        self.Lb1.pack()

        for i, s in enumerate(self._file1_keys):
            self.Lb1.insert(END, s)

    def populate_file2(self):
        if self.Lb2 is not None:
            self.Lb2.destroy()

        for c in self._block_dict.keys():
            if self._block_dict[c].get():
                self._file2_keys.append(c)
            self._block_dict[c].set(False)

        self.Lb2 = Listbox(self.cleanFrame2, width=10, selectmode=MULTIPLE)
        self.Lb2.pack()

        for i, s in enumerate(self._file2_keys):
            self.Lb2.insert(END, s)

    def save(self):
        file1 = {}
        for h in self._file1_keys.keys():
            c = h.split('_')[0]
            num = h.split('_')[1] - 1
            file1[c] = self._filedata[num][c]

        file2 = {}
        for h in self._file2_keys.keys():
            c = h.split('_')[0]
            num = h.split('_')[1] - 1
            file2[c] = self._filedata[num][c]

        curdir = os.getcwd()
        svdir = os.path.join(curdir, "fixed_files")

        if not os.path.exists(svdir):
            os.makedirs(svdir)

        pickle.dump(file1, open(os.path.join(svdir, "file1_repaired.pkl"), "wb"))
        pickle.dump(file2, open(os.path.join(svdir, "file2_repaired.pkl"), "wb"))


if __name__ == "__main__":
    root = Tk()
    main = FileRepairGUI(root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()
