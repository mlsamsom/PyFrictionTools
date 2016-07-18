import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
from utilities import moving_average, perpendicular, tribocycle_finder, \
    prony_hold, prony_min_hold, remove_outlier_friction


class BiomomentumData(object):
    """
    Author: Michael Samsom
    Description: An object that contains raw and parsed BiomomentumData with basic
                 data visualization and exploration methods.
    Methods:
    """

    def __init__(self, data=None, file_info=None):
        """
        This object loads data in biomomentum text file and parses in the constructor
        :type file_info: dictionary
        :type data: object
        """
        self._data = data
        self._file_info = file_info

    @property
    def data(self):
        """
        Getter for raw data attribute
        :return:
        """
        # print("Getting data")
        return self._data

    @data.setter
    def data(self, value):
        """
        set raw data
        :type value: list
        :param value:
        :return:
        """

        # print("Setting data value")
        self._data = value

    @property
    def info(self):
        """
        Getter for file info
        :return:
        """
        return self._file_info

    @info.setter
    def info(self, value):
        """
        Sets file info parameter
        :param value:
        :return:
        """
        self._file_info = value

    def block_ids(self):
        """

        :return: list of ids in biomomentum data
        """
        ids = []
        for h in self._data.keys():
            ids.append(self._data[h]['tag'][0][1:-1])

        print 'found %s blocks' % len(ids)
        return ids

    def show_block(self, block, variable):
        """
        Plots a variable from a block from the parsed data
        :param variable: string 'force', 'disp', 'torque', 'time'
        :param block:
        """
        if type(block) != int:
            raise SystemExit('showBlock not used correctly')

        if variable != 'force' or 'disp' or 'torque' or 'time':
            raise SystemExit('invalid variable in show_block')

        h = 'block' + str(block)

        if variable == 'time':
            print 'Why do you want to look at a straight line?'
            plt.plot(self._data[h]['data'][variable][:])
            plt.xlabel(self._data[h]['header'][0])
            plt.show()

        elif variable == 'disp' or 'force' or 'torque':
            plt.subplot(3, 1, 1)
            plt.plot(self._data[h]['data'][variable][:, 0])
            plt.xlabel('X')
            plt.title(variable)

            plt.subplot(3, 1, 2)
            plt.plot(self._data[h]['data'][variable][:, 1])
            plt.xlabel('Y')

            plt.subplot(3, 1, 3)
            plt.plot(self._data[h]['data'][variable][:, 2])
            plt.xlabel('Z')

            plt.show()

        else:
            raise SystemExit('showBlock(int, string) not used correctly')

    def numpy_array_parse(self, tag='<Scripts>', variable='disp'):
        """
        parses data of interest and compiles it into a list of numpy arrays
        where the list index corresponds to the block number
        :param tag: valid tag name eg '<Scripts>'
        :param variable: 'time', 'disp', 'force', 'torque'
        :return: list of ndarrays
        """
        out_array = []
        for h in self._data.keys():
            if self._data[h]['tag'] == [tag]:
                out_array.append(self._data[h]['data'][variable][:])

        return out_array


class FrictionData(BiomomentumData):
    """
    This object
    """
    area = 1e-5
    sample_run = 0  # use first friction run as sample to get friction run

    def _get_analysis_frame(self):
        """
        Gets the start and end of the analysis file
        :rtype: int
        :return: st, mid, end
        """
        st = []
        end = []
        mid = []
        # get analysis windows (just use first one)
        for h in self._data.keys():
            if self._data[h]['tag'] == ['<Scripts>']:
                cycle_ends, cycle_mid = tribocycle_finder(self._data[h]['data']['disp'][:, 1])
                break

        if 'cycle_mid' in locals():
            for c in xrange(cycle_mid[0].size):
                st.append(cycle_ends[0][c])
                end.append(cycle_ends[0][c + 1])
                mid.append(cycle_mid[0][c])
        else:
            raise SystemExit("tribocyclefinder did not run")

        c = self.sample_run  # the friction cycle to use
        if mid[c] - st[c] > end[c] - mid[c]:
            end[c] += 1
        elif mid[c] - st[c] < end[c] - mid[c]:
            st[c] -= 1

        return st, mid, end

    def _pack_force_curves(self):
        """
        DESCRIPTION: Prepares force readouts for analysis, projects the Y-Z
        into normal-shear space.
        INPUT: GLOBAL data
        OUTPUT: GLOBAL N, F
        :rtype: numpy array
        """
        if self._data is None:
            raise SystemExit("You must set data to parsed Biomomentum data")

        # analyze all friction blocks
        F_n_pos = []
        F_n_neg = []
        F_f_pos = []
        F_f_neg = []

        # get analysis windows (just use first one)
        st, mid, end = self._get_analysis_frame()

        for h in self._data.keys():
            if self._data[h]['tag'] == ['<Scripts>']:
                # get center of circle in machine coordinate system
                cen_str = self._data[h]['info'][-3][0].split(',')
                # eliminate "
                try:
                    center = np.array([cen_str[1], cen_str[0][3:]], dtype=float)
                except ValueError:
                    cen_str[0] = cen_str[0].replace('"', '')
                    center = np.array([cen_str[1], cen_str[0][3:]], dtype=float)

                # the perdendicular vectors are just the y-z positions minus center
                perp = self._data[h]['data']['disp'][:, 1:]
                perp -= center * np.ones(perp.shape)

                # normalize perpendicular vector
                perp /= np.linalg.norm(perp, axis=1)[:, None]

                # get parallel array
                par = perpendicular(perp)

                F_yz = self._data[h]['data']['force'][:, 1:]

                F_f_v = (par[:, 0] * F_yz[:, 0] + par[:, 1] * F_yz[:, 1]).T * F_yz.T
                F_f_v = F_f_v.T
                F_f = np.linalg.norm(F_f_v, axis=1)

                F_n_v = (perp[:, 0] * F_yz[:, 0] + perp[:, 1] * F_yz[:, 1]).T * F_yz.T
                F_n_v = F_n_v.T
                F_n = np.linalg.norm(F_n_v, axis=1)

                c = self.sample_run
                F_n_pos.append(F_n[st[c]:mid[c]])
                F_n_neg.append(F_n[mid[c]:end[c]])
                F_f_pos.append(F_f[st[c]:mid[c]])
                F_f_neg.append(F_f[mid[c]:end[c]])

        shape = (len(F_n_pos), F_n_pos[0].size, 2)

        self.N_all = np.zeros(shape, dtype=float)
        self.F_all = np.zeros(shape, dtype=float)

        for i in xrange(shape[0]):
            self.N_all[i, :, 0] = F_n_pos[i]
            self.N_all[i, :, 1] = F_n_neg[i]
            self.F_all[i, :, 0] = F_f_pos[i]
            self.F_all[i, :, 1] = F_f_neg[i]

    def friction_frame(self):
        """

        :return: self.fric_frame
        """
        self._pack_force_curves()

        if self._file_info is None:
            raise SystemExit("You must set info to a decomposed dictionary (see decompose_file_name in filehandling")

        shp = self.N_all.shape
        mid = int(np.floor(shp[1] / 2))
        rows = len(range(mid, shp[1]))

        length = rows * shp[0] * 2

        col_names = ['N', 'F', 'load_index', 'direction']
        pos = np.zeros(rows)
        neg = np.ones(rows)

        # initialize
        i = 0
        temp = np.zeros((rows, 4))
        temp2 = np.zeros((rows, 4))
        # positive
        temp[:, 0] = self.N_all[i, mid:, 0]
        temp[:, 1] = self.F_all[i, mid:, 0]
        temp[:, 2] = neg * i
        temp[:, 3] = pos
        # negative
        temp2[:, 0] = self.N_all[i, mid:, 1]
        temp2[:, 1] = self.F_all[i, mid:, 1]
        temp2[:, 2] = neg * i
        temp2[:, 3] = neg

        nparray = np.vstack((temp, temp2))

        for i in xrange(1, shp[0]):
            # positive
            temp[:, 0] = self.N_all[i, mid:, 0]
            temp[:, 1] = self.F_all[i, mid:, 0]
            temp[:, 2] = neg * i
            temp[:, 3] = pos
            # negative
            temp2[:, 0] = self.N_all[i, mid:, 1]
            temp2[:, 1] = self.F_all[i, mid:, 1]
            temp2[:, 2] = neg * i
            temp2[:, 3] = neg

            nparray = np.vstack((nparray, temp, temp2))

        index = self._file_info['name'][0:6] + '_' + str(self._file_info['rpt'])
        self.fric_frame = pd.DataFrame(nparray)
        self.fric_frame.index = [index] * length
        self.fric_frame.columns = col_names
        self.fric_frame['lens'] = [self._file_info['condition']] * length

        return self.fric_frame

    def prony_frame(self):
        """
        DESCRIPTION: Does a simple Prony series analysis on the stress relax
        data before each run. Outputs fits from all runs.
        INPUT: BOOLEAN plot
        OUTPUT: GLOBAL LIST popt_all
        """
        self.Fz_all = []
        rz_all = []
        self.t1_all = []
        popt_all = []
        self.t_all = []
        self.e1_all = []
        E_all = []

        for h in self._data.keys():
            if self._data[h]['tag'] == ['<Stress Relaxation>']:
                Fz_r = self._data[h]['data']['force'][:, 2]  # Pa
                rz_r = self._data[h]['data']['disp'][:, 2]
                t_r = self._data[h]['data']['time']

                # Calculate instantaneous modulus
                #  pull out strain rate
                strain_amp = float(self._data[h]['info'][14][1])  # mm
                strain_rate = float(self._data[h]['info'][15][1])  # mm/s

                # calculate strain assuming thickness of 1.5 mm
                strain = strain_amp / 1.5

                # start of relaxation test

                # find end of relaxation time
                Fz_grad_filt = moving_average(np.gradient(Fz_r))
                st = np.argmin(Fz_grad_filt)

                Fz = np.absolute(Fz_r[st:]) / self.area
                t = t_r[st:]
                rz = rz_r[st:]

                self.Fz_all.append(Fz)
                rz_all.append(rz)
                self.t_all.append(t)

                t1 = t_r[st]
                e1 = strain

                E_est = Fz[0] / e1 / t1

                x1 = np.array([0.2, 1.])
                bounds1 = [(0.0001, None), (0.01, 1.)]
                x, f, d = fmin_l_bfgs_b(prony_min_hold, x0=x1, args=(t, Fz, t1, e1, E_est),
                                        bounds=bounds1, approx_grad=True)

                popt_all.append(x)
                self.t1_all.append(t1)
                self.e1_all.append(e1)
                E_all.append(E_est)

            E_inst = E_all
            prony_const = np.array(popt_all)[:, 0]
            tau = np.array(popt_all)[:, 1]

            E_vec = np.empty(len(E_inst) * 2, dtype=float)
            p_vec = np.empty(len(prony_const) * 2, dtype=float)
            tau_vec = np.empty(len(tau) * 2, dtype=float)

            E_vec[0::2] = E_inst
            E_vec[1::2] = E_inst
            p_vec[0::2] = prony_const
            p_vec[1::2] = prony_const
            tau_vec[0::2] = tau
            tau_vec[1::2] = tau

            self.prony_out_frame = pd.DataFrame({'E': E_vec, 'p': p_vec, 'tau': tau_vec})

        return self.prony_out_frame

    def alignment_frame(self):
        """
        Returns a data frame containing the mean position of the "center of shear"
        during the friction test.
        :return: alig_frame (data frame)
        """
        st = []
        end = []
        mid = []

        align_pos = []
        align_neg = []
        mean_align_pos = []
        mean_align_neg = []

        # get analysis windows (just use first one)
        st, mid, end = self._get_analysis_frame()

        c = self.sample_run

        for h in self.data.keys():
            if self.data[h]['tag'] == ['<Scripts>']:
                # get center of circle in machine coordinate system
                cen_str = self.data[h]['info'][-3][0].split(',')
                # eliminate "
                try:
                    center = np.array([cen_str[1], cen_str[0][3:]], dtype=float)
                except ValueError:
                    cen_str[0] = cen_str[0].replace('"', '')
                    center = np.array([cen_str[1], cen_str[0][3:]], dtype=float)

                # positive sliding direction
                F_pos = self.data[h]['data']['force'][st[c]:mid[c], :]
                T_pos = self.data[h]['data']['torque'][st[c]:mid[c], 0:2]
                T_pos[:, [0, 1]] = T_pos[:, [1, 0]]  # swap x,y for calc

                # negative sliding direction
                F_neg = self.data[h]['data']['force'][mid[c]:end[c], :]
                T_neg = self.data[h]['data']['torque'][mid[c]:end[c], 0:2]
                T_neg[:, [0, 1]] = T_neg[:, [1, 0]]

                # calculate misalignment vector
                align_vec_pos = (T_pos - 30. * F_pos[:, 0:2])
                align_vec_pos[:, 0] /= F_pos[:, 2]
                align_vec_pos[:, 1] /= F_pos[:, 2]
                align_vec_neg = (T_neg - 30. * F_neg[:, 0:2])
                align_vec_neg[:, 0] /= F_neg[:, 2]
                align_vec_neg[:, 1] /= F_neg[:, 2]

                mean_pos = np.mean(align_vec_pos, axis=0)
                mean_neg = np.mean(align_vec_neg, axis=0)

                align_pos.append(align_vec_pos)
                align_neg.append(align_vec_neg)
                mean_align_pos = np.concatenate((mean_align_pos, mean_pos), axis=0)
                mean_align_neg = np.concatenate((mean_align_neg, mean_neg), axis=0)

        alig_p = mean_align_pos / 10.
        alig_n = mean_align_neg / 10.

        alig_p_vec = alig_p.reshape(alig_p.size / 2, 2)
        alig_n_vec = alig_n.reshape(alig_n.size / 2, 2)

        alig_vec = np.empty((alig_p.size, 2), dtype=alig_p_vec.dtype)
        alig_vec[0::2, :] = alig_p_vec
        alig_vec[1::2, :] = alig_n_vec

        self.alig_frame = pd.DataFrame(alig_vec)
        self.alig_frame.columns = ['alig_x', 'alig_y']

        return self.alig_frame

    def mean_friction_frame(self):
        """
        Creates a pandas data frame with mean friction data
        :return: self.mean_fric_frame
        """
        self.friction_frame()

        mean_ff = self.fric_frame.groupby(['load_index', 'direction']).mean()
        std_ff = self.fric_frame.groupby(['load_index', 'direction']).std()

        mean_ff = mean_ff.reset_index()

        mean_ff['Nstd'] = std_ff['N'].values
        mean_ff['Fstd'] = std_ff['F'].values

        p_tmp = mean_ff[mean_ff['direction'] == 0]
        n_tmp = mean_ff[mean_ff['direction'] == 1]

        # outlier test
        mu_p, oli_p = remove_outlier_friction(p_tmp['N'].values, p_tmp['F'].values)
        mu_n, oli_n = remove_outlier_friction(n_tmp['N'].values, n_tmp['F'].values)

        mean_ff['mu'] = np.array([mu_p, mu_n] * len(p_tmp))

        self.mean_fric_frame = mean_ff

        return self.mean_fric_frame

    def test_summary_frame(self):
        """

        :rtype: DataFrame
        :return: self.summary_frame
        """
        self.mean_friction_frame()
        self.alignment_frame()
        self.prony_frame()

        self.summary_frame = pd.concat([self.mean_fric_frame,
                                        self.alig_frame,
                                        self.prony_out_frame], axis=1)
        self.summary_frame['lens'] = [self._file_info['condition']] * len(self.summary_frame)
        self.summary_frame['test'] = self._file_info['name'][0:6] + '_' + str(self._file_info['rpt'])

        return self.summary_frame

    def excel_frame(self):
        """
        Returns a data frame for use in excel
        this is a relic from older analysis methods and is not recommended.

        Note excel_out_frame is not a member of this instance, only use as function
        :return: excel_out_frame
        """
        self.mean_friction_frame()

        tmp_df_p = self.mean_fric_frame[self.mean_fric_frame['direction'] == 0][['N', 'F']]
        tmp_df_p.index = [self._file_info['condition']] * len(tmp_df_p)
        tmp_df_p.columns = ['N(+)', 'F(+)']
        tmp_df_n = self.mean_fric_frame[self.mean_fric_frame['direction'] == 1][['N', 'F']]
        tmp_df_n.index = [self._file_info['condition']] * len(tmp_df_n)
        tmp_df_n.columns = ['N(-)', 'F(-)']

        excel_out_frame = pd.concat([tmp_df_p, tmp_df_n], axis=1)

        return excel_out_frame

    def show_projection(self, run=0, sns_context="talk"):
        print "untested"
        self.friction_frame()

        pos_dat = self.fric_frame[self.fric_frame['direction'] == 0]
        neg_dat = self.fric_frame[self.fric_frame['direction'] == 1]

        sns.set_context(sns_context)

        f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=2, ncols=2)
        pos_F = pos_dat[pos_dat['load_index'] == run]
        neg_F = neg_dat[neg_dat['load_index'] == run]

        ax1.plot(pos_F['N'].values)
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("(+)Normal force (N)")

        ax2.plot(pos_F['F'].values)
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("(+)Shear force (N)")

        ax3.plot(neg_F['N'].values)
        ax3.set_xlabel("Samples")
        ax3.set_ylabel("(-)Normal force (N)")

        ax4.plot(neg_F['F'].values)
        ax4.set_xlabel("Samples")
        ax4.set_ylabel("(-)Shear force (N)")

        plt.show()

    def show_friction_curve(self, run=0, sns_context="talk"):
        """
        Shows the tribological curves
        :param sns_context:
        :param run:
        :return: None
        """
        self.friction_frame()

        pos_dat = self.fric_frame[self.fric_frame['direction'] == 0]
        neg_dat = self.fric_frame[self.fric_frame['direction'] == 1]

        # plot formatting
        sns.set_context(sns_context)
        p_curve = sns.jointplot(x="N", y="F", data=pos_dat[pos_dat['load_index'] == run])
        p_curve.set_axis_labels(xlabel="Perpendicular Force (N)",
                                ylabel="Tangential Force (N)")
        p_curve.fig.suptitle("Positive direction")

        n_curve = sns.jointplot(x="N", y="F", data=neg_dat[neg_dat['load_index'] == run])
        n_curve.set_axis_labels(xlabel="Perpendicular Force (N)",
                                ylabel="Tangential Force (N)")
        n_curve.fig.suptitle("Negative direction")
        plt.show()

    def show_mean_alignment(self, run=0):
        """

        :param run: int integer corresponding to block of interest
        :return: None
        """
        self.test_summary_frame()

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        x = self.summary_frame[self.summary_frame['load_index'] == run]['alig_x'].values
        y = self.summary_frame[self.summary_frame['load_index'] == run]['alig_y'].values

        ax1.plot(x[0], y[0], "or")
        ax1.set_xlabel("x position (mm)")
        ax1.set_ylabel("y position (mm)")

        ax2.plot(x[1], y[1], "or")
        ax2.set_xlabel("x position (mm)")
        ax2.set_ylabel("y position (mm)")

        plt.show()

    def show_prony_fit(self, run=0, sns_context="talk"):
        self.test_summary_frame()

        F = self.Fz_all[run]

        E = self.summary_frame.loc[run]['E']
        p = self.summary_frame.loc[run]['p']
        tau = self.summary_frame.loc[run]['tau']
        e1 = self.e1_all[run]
        t1 = self.t1_all[run]
        t = self.t_all[run]

        y = prony_hold(t, E, p, tau, e1, t1)

        f, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(t, F, 'o')
        ax.hold(True)
        ax.plot(t, y)

        plt.show()

    def show_friction_line(self, sns_context="talk"):
        """
        Shows results from dF/dN friction test
        :param sns_context:
        :return: None
        """
        self.mean_friction_frame()

        p_dat = self.mean_fric_frame[self.mean_fric_frame['direction'] == 0]
        n_dat = self.mean_fric_frame[self.mean_fric_frame['direction'] == 1]

        xbins = self.mean_fric_frame['load_index'].max() + 1

        sns.set_context(sns_context)

        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        sns.regplot(x='N', y='F', data=p_dat, x_bins=xbins, ax=ax1)
        sns.regplot(x='N', y='F', data=n_dat, x_bins=xbins, ax=ax2)

        ax1.set_title("(+)")
        ax2.set_title("(-)")

        plt.tight_layout()

        plt.show()
