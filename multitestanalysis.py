import pandas as pd
import os
import filehandling as fh
import testanalysis as ta
import frictionstats as fs
from scipy import stats


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def concat_dfs(directory, save=True):
    """
    Concatenates all the picked dataframes in a given directory
    :param directory:
    :param save:
    :return:
    """
    pickle_list = os.listdir(directory)

    df = pd.read_pickle(os.path.join(directory, pickle_list[0]))

    name = pickle_list[0].split('_')[0]

    for f in pickle_list[1:]:
        temp = pd.read_pickle(os.path.join(directory, f))
        df = pd.concat([df, temp])

    if save:
        df.to_pickle(name)

    return df


valid_dirs_list = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
                   "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def analyze_multi(parent_dir, save=True, valid_dirs=valid_dirs_list):
    """
    Analyzes all tests in a given directory assuming roman numeral
    naming convention.
    :param save: Boolean
    :param valid_dirs: list
    :param parent_dir:string
    :return:
    """
    print "opening directory: ", parent_dir

    children_dir_unfilt = get_immediate_subdirectories(parent_dir)

    children_dir = list(set(children_dir_unfilt) & set(valid_dirs))

    if children_dir == set([]):
        print("Note: This program only detects directories up to 20, update valid_dirs to change")
        raise SystemExit("There are no valid directory names in your parent directory")

    if 'pickle' in children_dir:
        print("Warning: some analysis has been done here")

    for i, name in enumerate(children_dir):
        children_dir[i] = os.path.join(name, parent_dir)

    directory = children_dir[0]
    files = os.listdir(directory)
    data_objs = ta.get_friction_data_objects(files)
    big_df = ta.full_test_summary_frame(data_objs)
    for directory in children_dir[1:]:
        files = os.listdir(directory)
        data_objs = ta.get_friction_data_objects(files)
        temp_df = ta.full_test_summary_frame(data_objs)
        big_df = pd.concat([big_df, temp_df])

    name = os.path.basename(parent_dir)
    if save:
        big_df.to_pickle(name+"_DataFrame")

    return big_df


def friction_stats(df, effect_variable='mu'):
    """
    Builds a statistical report for a data frame
    """
    pass


if __name__ == '__main__':
    # par_dir = fh.get_dir()
    par_dir = '/home/michael/Dropbox/Eyelid_edge_dev/Test_Data/BiomomentumData/TestData/MS1604'

    df = analyze_multi(parent_dir=par_dir, save=False)
    print(df)
    # repeated measures variables

    # Figure out n and get descriptive statistics

    # Friedman chi square test
    measurements = []
    num_lenses = None
    fried_stats, fried_ps = stats.friedmanchisquare()







