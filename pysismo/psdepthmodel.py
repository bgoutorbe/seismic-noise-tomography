"""
Module taking care of the forward modelling: theoretical dispersion
curve given a 1D crustal model of velocities and densities.
Uses the binaries of the Computer Programs in Seismology, with
must be installed in *COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR*
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import itertools as it
from easyprocess import EasyProcess
import tempfile

# getting the dir of the binaries of the Computer Programs in Seismology
from psconfig import COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR


# default header of the model file:
# isotropic, 1D, flat Earth with layers of constant velocity
MODEL_HEADER = """MODEL.01
TEST
ISOTROPIC
KGS
FLAT EARTH
1-D
CONSTANT VELOCITY
LINE08
LINE09
LINE10
LINE11
H   VP  VS RHO  QP  QS ETAP ETAS FREFP FREFS"""


class VsModel:
    """
    Class holding a layered model of Vs function of depth,
    with Vp/Vs and rho/Vs ratio fixed.
    """
    def __init__(self, vs, dz, ratio_vp_vs, ratio_rho_vs, name='',
                 store_vg_at_periods=None):
        """
        Initializes model with layers' Vs (vs), layers' thickness (dz),
        and layers' ratio Vp/Vs and rho/Vs (ratio_vp_vs, ratio_rho_vs).
        """
        # checking shapes
        nlayers = np.size(vs)
        if np.size(dz) != nlayers - 1:
            raise Exception("Size of dz should be nb of layers minus 1")
        if not np.size(ratio_vp_vs) in [1, nlayers]:
            raise Exception("Size of ratio_vp_vs should be nb of layers or 1")
        if not np.size(ratio_rho_vs) in [1, nlayers]:
            raise Exception("Size of ratio_rho_vs should be nb of layers or 1")

        self.name = name
        self.vs = np.array(vs)
        self.dz = np.array(dz)
        self.ratio_vp_vs = np.array(ratio_vp_vs)
        self.ratio_rho_vs = np.array(ratio_rho_vs)

        # storing vg model at selected periods if required
        self.stored_vgperiods = store_vg_at_periods
        if not store_vg_at_periods is None:
            self.stored_vg = self.vg_model(store_vg_at_periods)
        else:
            self.stored_vg = None

    def misfit_to_vg(self, periods, vg, sigmavg, squared=True,
                     use_storedvg=True, storevg=False):
        """
        Misfit of modelled vg to observed vg

              [vg_model - vg]**2
        = Sum ------------------  over periods
                2 x sigmavg**2
        """
        # using stored vg model if required and available, else re-calculating it
        if use_storedvg and np.all(periods == self.stored_vgperiods):
            vg_model = self.stored_vg
        else:
            vg_model = self.vg_model(periods, store=storevg)

        misfit = np.sum(((vg_model - vg) / sigmavg)**2) / 2.0
        if squared:
            misfit = np.sqrt(misfit)
        return misfit

    def vg_model(self, periods, store=False):
        """
        Modelled group velocities, vg, function of period
        """
        vs = self.vs
        vp = self.ratio_vp_vs * self.vs
        rho = self.ratio_rho_vs * self.vs
        dz = np.r_[self.dz, 0]  # we append a fake thickness
        vg = Rayleigh_group_velocities(periods, dz=dz, vp=vp, vs=vs, rho=rho)
        if store:
            # storing group velocities if required
            self.stored_vgperiods = periods
            self.stored_vg = vg
        return vg

    def get_vs_at(self, z):
        """
        Returns Vs ad depth(s) *z*
        """
        indices = np.searchsorted(np.r_[0, self.dz.cumsum()], z, side='right') - 1
        if np.any(indices) < 0:
            raise Exception("Depth out of range")
        return self.vs[indices]

    def plot(self, periods, obsvgarrays=None, fig=None, color='r'):
        """
        Plots modelled and observed group velocity function of period (top)
        and the model itself, i.e. Vs vs depth (bottom)
        """
        if not fig:
            fig = plt.figure(figsize=(6.5, 10), tight_layout=True)
            axlist = [fig.add_subplot(211), fig.add_subplot(212)]
            legend = True
        else:
            axlist = fig.get_axes()
            legend = False  # no need to add legend to existing fig

        # 1st subplot: group velocity vs period
        ax = axlist[0]
        self.plot_vg(periods, obsvgarrays=obsvgarrays, ax=ax, legend=legend, color=color)
        ax.set_title(self.name)

        # 2nd subplot: Vs vs depth
        ax = axlist[1]
        self.plot_model(ax=ax, color=color)

        fig.canvas.draw()
        fig.show()
        return fig

    def plot_vg(self, periods, obsvgarrays=None, ax=None, legend=True, color='r'):
        """
        Plots modelled and observed group velocity function of period
        """
        # creating figure if not given as input
        fig = None
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        vg_model = self.vg_model(periods)
        ax.plot(periods, vg_model, lw=1.5, color=color, label=self.name)
        if obsvgarrays:
            for i, vgarray in enumerate(obsvgarrays):
                label = 'Observed dispersion curves' if not i else None
                ax.plot(periods, vgarray, lw=0.5, color='k', label=label)
        ax.set_xlabel('Period (sec)')
        ax.set_ylabel('Group velocity (km/s)')
        if legend:
            ax.legend(loc='best', fontsize=11, framealpha=0.8)
        ax.grid(True)

        if fig:
            fig.show()

    def plot_model(self, ax=None, color='r', format_axes=True):
        """
        Plots the model, i.e. Vs vs depth
        """
        # creating figure if not given as input
        fig = None
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        x = list(it.chain.from_iterable([[v, v] for v in self.vs]))
        y = [0.0] + list(it.chain.from_iterable([[z, z] for z in np.cumsum(self.dz)])) + \
            [self.dz.sum() + 15]
        ax.plot(x, y, lw=1.5, color=color)

        if format_axes:
            ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
            ax.set_xlabel('Vs (km/s)')
            ax.set_ylabel('Depth (km)')
            ax.grid(True)

        if fig:
            fig.show()


def Rayleigh_group_velocities(periods, dz, vp, vs, rho, verbose=False):
    """
    Returns the array of Rayleigh wave group velocities at
    selected periods, from the 1-D layered Earth model
    contained in *dz* (thicknesses), *vp* (P wave velocities),
    *vs* (S wave velocities) and *rho* (densities).

    The Computer Programs in Seismology, located in dir
    *COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR*,
    are used for the computation.
    """
    if not COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR:
        raise Exception("Please provide the dir of the Computer Programs in Seismology")

    # making and moving to temporary dir
    current_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)

    # preparing input files
    if verbose:
        print 'Preparing model and periods files'
    create_model_file('model', dz, vp, vs, rho)
    f = open('periods', 'w')
    f.write('\n'.join([str(p) for p in periods]))
    f.close()

    # preparing model
    if verbose:
        print "Calling sprep96"
    cmd = os.path.join(COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR, 'sprep96')
    # Rayleigh wave, fundamental mode
    p = EasyProcess('"{}" -M model -PARR periods -NMOD 1 -R'.format(cmd)).call()
    if verbose:
        print p.stdout

    # phase dispersion curve
    if verbose:
        print "Calling sdisp96"
    cmd = os.path.join(COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR, 'sdisp96')
    p = EasyProcess('"{}" -v'.format(cmd)).call()
    if verbose:
        print p.stdout

    # group dispersion curve
    if verbose:
        print "Calling sregn96"
    cmd = os.path.join(COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR, 'sregn96')
    p = EasyProcess('"{}"'.format(cmd)).call()
    if verbose:
        print p.stdout

    # exporting group velocities (-U) of Rayleigh waves (-R) in ascii file
    if verbose:
        print "Calling sdpegn96"
    cmd = os.path.join(COMPUTER_PROGRAMS_IN_SEISMOLOGY_DIR, 'sdpegn96')
    p = EasyProcess('"{}" -R -S -U -XLOG -PER -ASC'.format(cmd)).call()
    if verbose:
        print p.stdout

    # loading group velocities from 6th column of ascii file
    vg = np.loadtxt('SREGN.ASC', skiprows=1, usecols=(5,))

    # removing temp dir
    os.chdir(current_dir)
    shutil.rmtree(tmp_dir)

    return vg


def create_model_file(path, dz, vp, vs, rho):
    """
    Writing the 1D model to ascci file, to be used as input
    by the Computer Programs in Seismology
    """
    qp = np.zeros_like(dz)
    qs = np.zeros_like(dz)
    etap = np.zeros_like(dz)
    etas = np.zeros_like(dz)
    frefp = np.ones_like(dz)
    frefs = np.ones_like(dz)

    f = open(path, mode='w')
    f.write(MODEL_HEADER)

    a = np.vstack((dz, vp, vs, rho, qp, qs, etap, etas, frefp, frefs))
    for col in a.T:
        f.write('\n')
        col.tofile(f, sep=' ')

    f.close()