"""
This module contains Tkinter file and directory GUIs

"""
from Tkinter import *
import tkFileDialog
from tkFileDialog import askopenfilename
import os
from collections import OrderedDict
import numpy as np
import cPickle


def get_filename():
    """
    DESCRIPTION: Spawns a GUI window to ask user for a filename
    INPUT: None
    OUTPUT: fname (full path and filename of data file)
    spawns GUI file explorer window to choose data file

    EXAMPLE:
    fname = get_filename()
    :param message: Message to print in file dialogue
    """
    nottext = True
    count = 0
    while nottext:

        if count > 1000:
            print 'function is stuck'
            nottext = False

        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        fname = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

        if fname == u'':
            nottext = False
            raise SystemExit('Goodbye')

        if fname[-4:] == '.txt':
            nottext = False
        else:
            print 'WARNING: file chosen is not a text file'

        count += 1

    assert isinstance(fname, object)
    return fname


def get_dir(message="Select a directory"):
    """
    DESCRIPTION: Spawns a GUI window to ask user for a directory
    INPUT: None
    OUTPUT: list of files
    spawns GUI file explorer window to choose data file

    EXAMPLE:
    file_list = getFolderList()
    :param message: Message to print in file dialogue
    :param folder_type: str default "friction"
    """

    Tk().withdraw()
    directory = tkFileDialog.askdirectory(initialdir=os.getcwd(),
                                          title=message)

    return directory


def get_files_from_dir(directory, folder_type="friction"):
    """
    Gets all the friction files from a given directory
    :param directory: str
    :return: text_files
    """
    os.chdir(directory)

    list_of_files = os.listdir(directory)
    list_of_files.sort()

    text_files = []

    for i in xrange(len(list_of_files)):
        istext = list_of_files[i][-4:] == '.txt'
        isfriction = False
        if folder_type in list_of_files[i]:
            isfriction = True

        if istext and isfriction:
            text_files.append(directory + '/' + list_of_files[i])

    return text_files


def parse_file(fn):
    """
    DESCRIPTION: This parses a Mach-1 text file by block into a python dictionary object
    INPUT: STRING fname (full path and filename of data file)
    OUTPUT: DICTIONARY data, all data organized in the following way

                             block
                    ___________|____________RAW
                    |           |      |    |
                  data       header   info  tag
            ________|_________
            |    |     |     |
          disp  force time torque

    header is the data variable header
    info is the system info at the top of each block
    tag is the name of the function of each block
    data is the data in numpy array form. disp, force and torque are 3D vectors (x,y,z), time is a 1D vector

    EXAMPLE:
    data = parse_file(fname)
    :param fn: String
    :rtype: parsed data in OrderedDict

    """

    f = open(fn, 'r')
    _raw = []
    # read all data into a list
    for row in f:
        _raw.append(row.strip().split('\t'))

    f.close()

    # Check if it's a Mach-1 file
    if _raw[0][0] != '<Mach-1 File>':
        raise SystemExit('File is not a Mach-1 output')

    _data = OrderedDict()

    # extract blocksRAW
    blkct = 0
    rec_info = False
    data_st = []
    data_end = []
    for row in xrange(len(_raw) - 1):
        # check for new block
        if _raw[row] == ['<INFO>']:
            rec_info = True
            blknm = 'block' + str(blkct)
            _data[blknm] = {'info': [],
                            'data': {'time': [],
                                     'disp': [],
                                     'force': [],
                                     'torque': []},
                            'tag': [],
                            'header': []}
            blkct += 1

        if _raw[row + 1] == ['<END INFO>']:
            _data[blknm]['tag'] = _raw[row + 2]

        if _raw[row + 1] == ['<DATA>']:
            rec_info = False

        if rec_info:
            _data[blknm]['info'].append(_raw[row + 1])

        # find data blocks for conversion to numpy arrays
        if _raw[row] == ['<DATA>']:
            _data[blknm]['header'] = _raw[row + 1]
            data_st.append(row + 2)
        elif _raw[row + 1] == ['<END DATA>']:
            data_end.append(row + 1)

    # convert data to numpy array and assign to keys
    if len(data_st) != len(data_end):
        print('error data start or end tag missing')

    for block in xrange(len(data_st)):
        # time
        if _raw[data_end[block] - 1] == ['<divider>']:
            data_vec = np.array(_raw[data_st[block]:data_end[block] - 1], dtype=float)
        else:
            data_vec = np.array(_raw[data_st[block]:data_end[block]], dtype=float)
        # rearrange displacement to be x,y,z from z,x,y
        disp = np.copy(data_vec[:, 1:4])
        disp[:, 0:2] = data_vec[:, 2:4]
        disp[:, 2] = data_vec[:, 1]

        _data['block' + str(block)]['data']['time'] = data_vec[:, 0]
        _data['block' + str(block)]['data']['disp'] = disp
        _data['block' + str(block)]['data']['force'] = data_vec[:, 4:7]
        _data['block' + str(block)]['data']['torque'] = data_vec[:, 7:10]

    return _data


def decompose_file_name(filename):
    """
    Decomposes standard Schmidt lab named files for Biomomentum
    friction files
    :param filename: string
    :return: dictionary
    """
    fn = str(filename)
    name = fn.split('/')[-1]  # pull out filename
    name = name[0:-4]  # remove file extension
    name_list = name.split('_')

    repaired = False
    if name_list[-1] == 'repaired':
        repaired = True
        rpt = rom_to_int(name_list[-2])
    else:
        rpt = rom_to_int(name_list[-1])

    if name_list[-2] != 'friction':
        test = name_list[1]
        condition = name_list[-2]  # will need to assign a numerical code to each lens type
    else:
        test = name_list[-2]
        condition = name_list[1]  # will need to assign a numerical code to each lens type

    experimenter = ord(name_list[0][0]) + ord(name_list[0][1])

    name_info = {'expnum': name_list[0],
                 'rpt': rpt,
                 'test': test,
                 'condition': condition,
                 'experimenter': experimenter,
                 'name': name}

    return name_info, repaired


def rom_to_int(string):
    """
    Converts roman numeral to integer ( only up 50)
    (case sensitive)
    :param string: a roman numeral in lower case
    """

    table = [['l', 50], ['xl', 40], ['x', 10], ['ix', 9], ['v', 5], ['iv', 4], ['i', 1]]
    returnint = 0
    for pair in table:

        continueyes = True

        while continueyes:
            if len(string) >= len(pair[0]):

                if string[0:len(pair[0])] == pair[0]:
                    returnint += pair[1]
                    string = string[len(pair[0]):]

                else:
                    continueyes = False
            else:
                continueyes = False

    return returnint


def repair_blocks():
    """
    moves blocks from one file to another
    :return:
    """
    dir1 = get_filename()
    dir2 = get_filename()
    file1 = parse_file(dir1)
    file2 = parse_file(dir2)

    name1, rep1 = decompose_file_name(dir1)
    name2, rep2 = decompose_file_name(dir2)

    svnm1 = name1['name'] + '_' + 'repaired'
    svnm2 = name2['name'] + '_' + 'repaired'

    # build {number:tag} dictionary for each file
    dict1 = OrderedDict()
    for h in file1.keys():
        dict1[h] = file1[h]['tag'][0][1:-1]

    dict2 = OrderedDict()
    for h in file2.keys():
        dict2[h] = file2[h]['tag'][0][1:-1]

    done = 'n'

    swap_list1 = []
    swap_list2 = []
    while done == 'n':
        print "Block list 1:", dict1
        print "Block list 2:", dict2

        input1 = False
        while not input1:
            swap_this = raw_input("Swap this block: ")
            swap_this = int(swap_this)
            if swap_this <= len(dict1):
                print "swapping ", swap_this
                input1 = True
                
            else:
                print "Invalid input"

        input2 = False
        while not input2:
            with_this = raw_input("with: ")
            with_this = int(with_this)
            if with_this <= len(dict2):
                print "with ", with_this
                input2 = True
            else:
                print "Invalid input"

        swap_list1.append('block' + str(swap_this))
        swap_list2.append('block' + str(with_this))
        done = raw_input("Are your done? y/n ")

    new_file1 = OrderedDict()
    new_file2 = OrderedDict()

    i = 0
    for h in file1.keys():
        if h not in swap_list1:
            new_file1[h] = file1[h]
        else:
            new_file1[h] = file2[swap_list2[i]]
            i += 1
            print i

    i = 0
    for h in file2.keys():
        if h not in swap_list2:
            new_file2[h] = file2[h]
        else:
            new_file2[h] = file1[swap_list1[i]]
            i += 1

    print "Saving repaired files in: ", os.getcwd()

    cPickle.dump(new_file1, open(svnm1, "wb"))
    cPickle.dump(new_file2, open(svnm2, "wb"))

    directory = os.path.dirname(dir1)
    new_dir = os.path.join(directory, 'broken_files')

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    destination1 = os.path.join(new_dir, os.path.basename(dir1))
    destination2 = os.path.join(new_dir, os.path.basename(dir2))

    os.rename(dir1, destination1)
    os.rename(dir2, destination2)






