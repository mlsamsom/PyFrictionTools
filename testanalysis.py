import filehandling as fh
import data
import pandas as pd
import os
import cPickle


def get_friction_data_objects(list_of_files):
    """
    Reads friction data in a given directory and compiles in into
    a list of data.FrictionData objects for analysis.
    :param list_of_files: a list of filepathexamples to analyse
    :return: data_obj_list
    """
    # create list of data objects
    data_obj_list = []

    for i, fn in enumerate(list_of_files):
        print "getting file", fn
        file_info, repaired = fh.decompose_file_name(fn)

        if repaired:
            dat = cPickle.load(open(fn, "rb"))
        else:
            dat = fh.parse_file(fn)

        obj = data.FrictionData()

        obj.data = dat
        obj.info = file_info

        data_obj_list.append(obj)

    return data_obj_list


def full_test_summary_frame(data_obj_list):
    """
    Compiles data from all tests into large data frame
    :param data_obj_list:
    :return: full_test_out_frame
    """
    full_test_out_frame = data_obj_list[0].test_summary_frame()

    for i, obj in enumerate(data_obj_list[1:]):
        print "analyzing file: ", obj._file_info['name']
        df = obj.test_summary_frame()
        full_test_out_frame = pd.concat([full_test_out_frame, df])

    return full_test_out_frame


def save_summary_frame(full_test_out_frame):
    """

    :param full_test_out_frame:
    :return:
    """

    parent_dir = os.path.dirname(os.getcwd())
    directory = os.path.join(parent_dir, 'pickles')
    svnm = full_test_out_frame['test'].values[0]
    svdir = os.path.join(directory, svnm)

    if not os.path.exists(directory):
        os.mkdir(directory)

    print "pickles saved in: ", directory

    full_test_out_frame.to_pickle(svdir)


def full_test_friction_frame(data_obj_list):
    """
    Compiles friction data from all tests into large data frame
    :param data_obj_list:
    :return: full_test_out_frame
    """
    full_test_out_frame = data_obj_list[0].friction_frame()

    for i, obj in enumerate(data_obj_list[1:]):
        df = obj.friction_frame()
        full_test_out_frame = pd.concat([full_test_out_frame, df])

    return full_test_out_frame


def excel_copy(file_list, save=True):
    """
    Description:
    Compiles friction data for an excel sheet analysis.
    This is a relic from the development phase in case anything needs to be
    replicated.
    :param save: Save frame as csv
    :param file_list: A list of file paths (usually from filehandling.get_files_from_dir()
    """
    # starter
    nm_dict, _ = fh.decompose_file_name(file_list[0])
    obj_list = get_friction_data_objects(file_list)

    df = obj_list[0].excel_frame()
    for obj in obj_list[1:]:
        print("analysing file {:s}".format(obj.info['name']))
        tmp = obj.excel_frame()
        df = pd.concat([df, tmp])

    if save:
        svnm = nm_dict['name'].split('_')[0] + '_' + str(nm_dict['rpt']) + '.csv'
        df.to_csv(svnm)
        print 'csv saved in', os.getcwd()

    return df
