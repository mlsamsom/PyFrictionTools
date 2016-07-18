""" June 22, 2016 Analysis script for Biomomentum Mach-1 Eyelid wiper friction test
    PyFrictionTools is python based analysis environment for mach-1 friction data
"""
from PyFrictionTools import filehandling, testanalysis, multitestanalysis
import os


def one_test():
    """
    Analyse a single repeat
    :return: None
    """
    # get directory with all the data files
    dn = filehandling.get_dir()

    # get a list of all friction files in the selected directory
    file_list = filehandling.get_files_from_dir(dn)

    # load all the data into FrictionData objects
    data_list = testanalysis.get_friction_data_objects(file_list)

    # All the data is analysed within these objects
    full_test_dataframe = testanalysis.full_test_summary_frame(data_list)
    testanalysis.save_summary_frame(full_test_dataframe) # save the data frame


def analyze_all_tests():
    """
    Analyze all test in a folder
    :return: None
    """
    # get directory with all the analysed files
    dn = filehandling.get_dir(message="full test directory")

    full_test_dataframe = multitestanalysis.analyze_multi(parent_dir=dn)
    print("saving data in", os.getcwd())

    # run stats on data

    # generate a report

