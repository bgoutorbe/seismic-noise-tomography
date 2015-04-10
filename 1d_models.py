#!/usr/bin/python -u
"""
This script inverts dispersion curves at selected locations
for a 1D shear velocity model.

The model is parametrized with:
- one sedimentary layer, with fixed Vs, Vp/Vs, rho/Vs and free thickness
- a user-defined number of basement layers, with fixed Vp/Vs, rho/Vs and
  free Vs and thickness
- a semi-infinite mantle with fixed Vs, Vp/Vs, rho/Vs
So, the model's parameters are (1) the thicknesses of the sediment and basement
layers and (2) the Vs of the basement layers. In order to define the space of
plausible parameters, you must provide bounds for these parameters, and you can
add additional constraints on (1) the minimum acceptable increment of Vs between
two adjacent layer (e.g., 0 to force an increasing Vs with depth) and (2) on the
acceptable range for the total crustal thickness (sediment + basement). For each
free parameter's bound and fixed property, you can give a default global value,
as well as location-specific values, in case you have at hand better constraints
from independant observations.

The Computer Programs in Seismology [Herrmann, R.B., 2013. Computer Programs in
Seismology: an evolving tool for instruction and research, Seismol. Res. Let., 84(6),
1081-1088] take care of the forward model, wherein a theoretical Rayleigh group
dispersion curve, vg_model(T), is calculated given a model. Observed dispersion
curves, vg_obs(T), are constructed from user-defined dispersion maps, by compiling
group velocities observed at a (user-defined) number of grid nodes near the selected
locations. The misfit to observed dispersion curves is then:

  S(m) = 1/2 sum{ [vg_model(T) - <vg_obs(T)>]^2 / sigma_obs(T)^2 } over T

with <vg_obs(T)> and sigma_obs(T) the mean and standard deviation of oberved
group velocities at period T, respectively.

The first step of the algorithm is to seek m0 that minimizes S(m) using a
constrained optimization (COBYLA). As the forward relationship is highly
non-linear, the optimization usually does a poor job, so this m0 shound not
be trusted as is: the objective is to provide a roughly acceptable starting
point to the Monte Carlo exploration that comes next.

The second step applies the Markov Chain Monte Carlo (MCMC) method of
Mosegaard, K. & Tarantola, A., [1995. Monte Carlo sampling of solutions to inverse
problems, J. Geophys. Res., 100(B7), 12431-12447] in order to sample the
posterior distribution of the parameters:

  f_post(m) = k.f_prior(m).L(m),

where f_prior is the prior distribution and L the likelihood function. The
algorithm uses Gaussian uncertainties, so that L(m) = exp(-S(m)). The prior
distribution is simply uniform within the parameters' bounds. It is sampled
with a prior random walk that, at each iteration:
- selects at random a thickness and a Vs,
- perturbates the selected thickness and Vs according to a uniform random walk,
  with user-defined step size an max jump size
This prior walk is then modified according to the Metropolis rule described by
Mosegaard & Tarantola [1995], in order to sample the posterior distribution.
Actually, the algorithm also rejects any perturbation towards an implausible
model (e.g., total crustal thickness out of bounds), which amounts to setting
a zero likelihood to implausible models and a uniform likelihood to plausible
models.

The results are exported to three files per location:
1) <location name> (prior distribution)_<suffix>.png
2) <location name>_<suffix>.png
3) <location name>_<suffix>.pickle
where <suffix> is a user-defined suffix.

The first file constains histogram plots of samples drawn from the prior
distribution, by accepting all the moves proposed by the prior walk.
You should check that you have drawn enough samples for the histograms to
look uniform.

The second file gives a summary of the posterior distribution of the
parameters. In particular, you'll find histograms of the sampled parameters
(and also of the depth of the base of the layers), with mean, standard dev,
95% confidence interval. Also shown are the observed and (95% interval of)
the theoretical dispersion curves and the (95% interval of) Vs vs depth,
together with the initial model (the one obtained after the optimization
step), the best-fitting model and the "representative" model (model closest
to the mean of posterior Vs vs depth).

The third file contains two objects exported with module pickle: (1) the
list of observed dispersion curves, as a list of numpy arrays
[vgarray1, vgarray2 ...], and (2) the models sampled from the posterior
distribution by the MCMC algorithm, as a list of instances of VsModel
(module psdepthmodel) [vsmodel1, vsmodel2 ...]
"""
from pysismo import psdepthmodel, psmcsampling
import os
import shutil
import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pysismo.psconfig import TOMO_DIR, DEPTHMODELS_DIR

# ==================================================================
# locations and associated names, {name: (lon, lat)}, around which
# extract the dispersion curves to be inverted. For each location,
# we extract the group velocities at the *NB_NEIGHBORS* nearest grid
# nodes of the dispersion maps
# ==================================================================

LOCATIONS = {'Parana basin': (-51.5, -22.5),
             'Sao Francisco craton': (-45.5, -18.5),
             'Tocantins province': (-48.5, -13.5)
             }
NB_NEIGHBORS = 4

print u"Select location(s) on which estimate depth models [all]:"
print '0 - All'
print '\n'.join('{} - {}'.format(i + 1, k) for i, k in enumerate(sorted(LOCATIONS)))
res = raw_input('\n')
if res:
    LOCATIONS = dict(sorted(LOCATIONS.items())[int(i) - 1] for i in res.split())

# ======================================================================
# parametrization of the model: number of crustal layers, ratio Vp/Vs,
# rho/Vs (of sediments, crust and mantle) and Vs of sediments and mantle
# ======================================================================

DEPTHS = np.arange(50)  # depths over which the model will be plotted

# sediments: default
VS_SEDIMENTS = 2.7
RATIO_VP_VS_SEDIMENTS = 1.815  # Vp = 4.9 km/s,    Snoke & James 1997, reported
RATIO_RHO_VS_SEDIMENTS = 0.93  # rho = 2.5 g/cm3   by Julia et al. 2008
# sediments: location-specific (where available)
LOCAL_VS_SEDIMENTS = {}
LOCAL_RATIO_VP_VS_SEDIMENTS = {}
LOCAL_RATIO_RHO_VS_SEDIMENTS = {}
# crust: defaut
NB_CRUST_LAYERS = 2
RATIO_VP_VS_CRUST = 1.745  # Chulick et al., 2013 (avg South America)
RATIO_RHO_VS_CRUST = 0.78  # avg rho/Vs for Vs=3.4-4, given the relationship rho(Vp)
                           # of Christensen & Mooney 1995 at 20 km and the Vp/Vs chosen
# crust: location-specific (where available)
LOCAL_NB_CRUST_LAYERS = {}
LOCAL_RATIO_VP_VS_CRUST = {}
LOCAL_RATIO_RHO_VS_CRUST = {}
# mantle: default
VS_MANTLE = 4.5              # Chulick et al., 2013 (avg South America)
RATIO_VP_VS_MANTLE = 1.778   # Vp = 8 km/s, Chulick et al. 2013 (avg South America)
RATIO_RHO_VS_MANTLE = 0.744  # rho = 3.35 g/cm3
# mante: location-specific (where available)
LOCAL_VS_MANTLE = {}
LOCAL_RATIO_VP_VS_MANTLE = {}
LOCAL_RATIO_RHO_VS_MANTLE = {}

# ==================================================================
# space of plausible models: bounds on layers' Vs, layers' thickness
# and Moho depth, minimum Vs increment between two layers
# ==================================================================

# default bounds
DZ_SEDIMENTS_BOUNDS = (0.0, 10.0)
VS_CRUST_BOUNDS = (3.2, 4.0)
DZ_CRUST_BOUNDS = (10.0, 30.0)
MOHO_DEPTH_BOUNDS = (35.0, 50.0)
# location-specific bounds (if available)
LOCAL_DZ_SEDIMENTS_BOUNDS = {'Parana basin': (2.5, 5.2),  # from Laske & Masters
                             'Sao Francisco craton': (0.0, 0.4),
                             'Tocantins province': (0.0, 0.1)}
LOCAL_VS_CRUST_BOUNDS = {}
LOCAL_DZ_CRUST_BOUNDS = {}
LOCAL_MOHO_DEPTH_BOUNDS = {'Parana basin': (39.0, 45.0),  # (42.0, 45.0),
                           'Sao Francisco craton': (37.0, 43.0),  # (38.0, 41.0),
                           'Tocantins province': (35.0, 41.0)  # (36.0, 44.0)
                           }   # from Assumpcao et al.
# min Vs increment (set to 0 to force Vs increase with depth)
VS_MIN_INCREMENT = 0.0

# =============================================================
# parameters of the Monte Carlo exploration: sampling step, max
# allowed jump, nb of samples and size of the burn in phase
# =============================================================

# crustal Vs
VS_SAMPLINGSTEP = 0.02
VS_MAXJUMP = 0.06
# thickness of sediment layer
DZ_SEDIMENTS_SAMPLINGSTEP = 0.1
DZ_SEDIMENTS_MAXJUMP = 0.5
# thickness of crustal layers
DZ_CRUSTLAYER_SAMPLINGSTEP = 1.0
DZ_CRUSTLAYER_MAXJUMP = 2.0
# nb of samples
NB_SAMPLES = 50000
res = raw_input('Number of samples of MC exploration? [{}]\n'.format(NB_SAMPLES))
NB_SAMPLES = int(res) if res.strip() else NB_SAMPLES
# nb of burnt in samples
NB_BURN = min(int(NB_SAMPLES / 10), 200)
res = raw_input('Number of burnt in samples? [{}]\n'.format(NB_BURN))
NB_BURN = int(res) if res.strip() else NB_BURN


# user-defined suffix to append to file names
usersuffix = raw_input("\nEnter suffix to append: [none]\n").strip()

# =====================================================================
# assigning to each period a dispersion map from which group velocities
# will be etxracted, in a dict {period: velocity map}
# =====================================================================

print "Loading velocity maps"
PICKLE_FILE_SHORT_PERIODS = os.path.join(
    TOMO_DIR, '2-pass-tomography_1996-2012_xmlresponse_3-60s_periods=6-10s.pickle')
PICKLE_FILE_LONG_PERIODS = os.path.join(
    TOMO_DIR, '2-pass-tomography_1996-2012_xmlresponse_7-60s_periods=10-30s.pickle')

with open(PICKLE_FILE_SHORT_PERIODS, 'rb') as f:
    VMAPS_SHORT = pickle.load(f)
with open(PICKLE_FILE_LONG_PERIODS, 'rb') as f:
    VMAPS_LONG = pickle.load(f)

PERIODVMAPS = {T: (VMAPS_SHORT[T] if T < 10 else VMAPS_LONG[T]) for T in range(6, 26)}
PERIODS = np.array(sorted(PERIODVMAPS.keys()))

# =========================
# loop on selected location
# =========================

for locname, (lon, lat) in sorted(LOCATIONS.items()):
    print "Working at location '{}': lon={}, lat={}".format(locname, lon, lat)

    # getting location-specific parameters/bounds if available, else default ones
    # parameters
    vs_sediments = LOCAL_VS_SEDIMENTS.get(locname, VS_SEDIMENTS)
    ratio_vp_vs_sediments = LOCAL_RATIO_VP_VS_SEDIMENTS.get(locname,
                                                            RATIO_VP_VS_SEDIMENTS)
    ratio_rho_vs_sediments = LOCAL_RATIO_RHO_VS_SEDIMENTS.get(locname,
                                                              RATIO_RHO_VS_SEDIMENTS)
    nb_crust_layers = LOCAL_NB_CRUST_LAYERS.get(locname, NB_CRUST_LAYERS)
    ratio_vp_vs_crust = LOCAL_RATIO_VP_VS_CRUST.get(locname, RATIO_VP_VS_CRUST)
    ratio_rho_vs_crust = LOCAL_RATIO_RHO_VS_CRUST.get(locname, RATIO_RHO_VS_CRUST)
    vs_mantle = LOCAL_VS_MANTLE.get(locname, VS_MANTLE)
    ratio_vp_vs_mantle = LOCAL_RATIO_VP_VS_MANTLE.get(locname, RATIO_VP_VS_MANTLE)
    ratio_rho_vs_mantle = LOCAL_RATIO_RHO_VS_MANTLE.get(locname, RATIO_RHO_VS_MANTLE)
    # arrays of ratio Vp/Vs and rho/Vs
    ratio_vp_vs = np.r_[ratio_vp_vs_sediments,
                        nb_crust_layers * [ratio_vp_vs_crust],
                        ratio_vp_vs_mantle]
    ratio_rho_vs = np.r_[ratio_rho_vs_sediments,
                         nb_crust_layers * [ratio_rho_vs_crust],
                         ratio_rho_vs_mantle]
    # bounds
    dz_sediments_bounds = LOCAL_DZ_SEDIMENTS_BOUNDS.get(locname, DZ_SEDIMENTS_BOUNDS)
    vs_crust_bounds = LOCAL_VS_CRUST_BOUNDS.get(locname, VS_CRUST_BOUNDS)
    dz_crust_bounds = LOCAL_DZ_CRUST_BOUNDS.get(locname, DZ_CRUST_BOUNDS)
    moho_depth_bounds = LOCAL_MOHO_DEPTH_BOUNDS.get(locname, MOHO_DEPTH_BOUNDS)

    # =================================================================
    # getting the dispersion curves at the *NB_NEIGHBORS* nearest nodes
    # =================================================================

    print "  getting observed dispersion curves at nearest nodes"
    vgarrays = [np.zeros_like(PERIODS, dtype='float') for _ in range(NB_NEIGHBORS)]
    meanvg = np.zeros_like(PERIODS, dtype='float')
    sigmavg = np.zeros_like(PERIODS, dtype='float')
    previous_grid = None
    previous_inodes = None
    for iT, T in enumerate(PERIODS):
        # getting the velocities of period T at the *NB_NEIGHBORS* nodes
        # closest to current location
        vmap = PERIODVMAPS[T]
        if vmap.grid == previous_grid:
            inodes = previous_inodes
        else:
            xnodes, ynodes = vmap.grid.xy_nodes()
            inodes = np.argsort((xnodes - lon)**2 + (ynodes - lat)**2)[:NB_NEIGHBORS]

        previous_grid = vmap.grid
        previous_inodes = inodes

        # velocities at period T + mean and std dev
        vels = np.array((vmap.v0 / (1 + vmap.mopt))).flatten()[inodes]
        for vg, vel in zip(vgarrays, vels):
            vg[iT] = vel
        meanvg[iT] = vels.mean()
        sigmavg[iT] = vels.std()

    # ==============================================================
    # quick estimate of best-fitting parameters (thicknesses and Vs)
    # ==============================================================

    def misfit(m):
        """
        Objective function to minimize: misfit between
        modelled and observed group velocities
        """
        # vector of parameters = [vs of crustal layers]
        #                        + [dz of sediment and crustal layers]
        vs_crust = m[:nb_crust_layers]

        # building the model
        vsmodel = psdepthmodel.VsModel(vs=np.r_[vs_sediments, vs_crust, vs_mantle],
                                       dz=m[nb_crust_layers:],
                                       ratio_vp_vs=ratio_vp_vs,
                                       ratio_rho_vs=ratio_rho_vs)
        return vsmodel.misfit_to_vg(periods=PERIODS,
                                    vg=meanvg,
                                    sigmavg=sigmavg)

    # estimating best-fitting parameters (thickness and Vs of layers)
    print "  estimating best-fitting depth model"

    # initial parameters
    dz0_sediments = 4.0
    vs0_crust = 3.7
    dz0_crust = (np.mean(moho_depth_bounds) - dz0_sediments) / nb_crust_layers
    vs0 = [vs0_crust] * nb_crust_layers
    dz0 = [dz0_sediments] + [dz0_crust] * nb_crust_layers
    m0 = np.array(vs0 + dz0)

    # constraint on Vs bounds
    constraints = []
    for i in range(nb_crust_layers):
        # Vs > min Vs
        constr = {'type': 'ineq', 'fun': lambda m, i=i: m[i] - vs_crust_bounds[0]}
        constraints.append(constr)
        # Vs < max Vs
        constr = {'type': 'ineq', 'fun': lambda m, i=i: vs_crust_bounds[1] - m[i]}
        constraints.append(constr)

    # constraints on sediment layer's thickness's bounds
    # thickness > min thickness
    constr = {'type': 'ineq',
              'fun': lambda m: m[nb_crust_layers] - dz_sediments_bounds[0]}
    constraints.append(constr)
    # thickness < min thickness
    constr = {'type': 'ineq',
              'fun': lambda m: dz_sediments_bounds[1] - m[nb_crust_layers]}
    constraints.append(constr)

    # constraints on crustal layer's thickness's bounds
    for i in range(nb_crust_layers + 1, len(m0)):
        # thickness > min thickness
        constr = {'type': 'ineq', 'fun': lambda m, i=i: m[i] - dz_crust_bounds[0]}
        constraints.append(constr)
        # thickness < min thickness
        constr = {'type': 'ineq', 'fun': lambda m, i=i: dz_crust_bounds[1] - m[i]}
        constraints.append(constr)

    # constraint on Moho depth
    constraints += [
        {'type': 'ineq',
         'fun': lambda m: np.sum(m[nb_crust_layers:]) - moho_depth_bounds[0]},
        {'type': 'ineq',
         'fun': lambda m: moho_depth_bounds[1] - np.sum(m[nb_crust_layers:])}
    ]

    # constraint on Vs increments:
    # Vs_nextlayer - Vs_currentlayer > VS_MIN_INCREMENT
    if not VS_MIN_INCREMENT is None:
        # constraint on increment between sediments and first crustal layer
        constr = {'type': 'ineq',
                  'fun': lambda m: m[0] - vs_sediments - VS_MIN_INCREMENT}
        constraints.append(constr)
        # constraints on increments between crustal layers
        for i in range(nb_crust_layers - 1):
            constr = {'type': 'ineq',
                      'fun': lambda m, i=i: m[i+1] - m[i] - VS_MIN_INCREMENT}
            constraints.append(constr)
        # constraint on increment between last crustal layer and mantle
        constr = {'type': 'ineq',
                  'fun': lambda m: vs_mantle - m[nb_crust_layers - 1] - VS_MIN_INCREMENT}
        constraints.append(constr)

    # constrained optimization
    mopt = minimize(misfit, x0=m0, method='COBYLA', constraints=constraints)['x']
    vscrustopt, dzsedopt, dzcrustopt = (mopt[:nb_crust_layers],
                                        mopt[nb_crust_layers],
                                        mopt[nb_crust_layers + 1:])

    # using the best-fitting model as initial model of the MC exploration
    vsmodelinit = psdepthmodel.VsModel(vs=np.r_[vs_sediments, vscrustopt, vs_mantle],
                                       dz=np.r_[dzsedopt, dzcrustopt],
                                       ratio_vp_vs=ratio_vp_vs,
                                       ratio_rho_vs=ratio_rho_vs,
                                       name='Initial model')

    # =======================
    # Monte-Carlo exploration
    # =======================

    # initializing parameters
    vscrustlayers = []
    dzcrustlayers = []
    for i in range(nb_crust_layers):
        vscrustlayer = psmcsampling.Parameter(
            name='Vs of crustal layer #{}'.format(i + 1),
            minval=vs_crust_bounds[0],
            maxval=vs_crust_bounds[1],
            step=VS_SAMPLINGSTEP,
            startval=vscrustopt[i],
            maxjumpsize=VS_MAXJUMP,
            nmaxsample=NB_SAMPLES)
        vscrustlayers.append(vscrustlayer)

        dzcrustlayer = psmcsampling.Parameter(
            name='Thickness of crustal layer #{}'.format(i + 1),
            minval=dz_crust_bounds[0],
            maxval=dz_crust_bounds[1],
            step=DZ_CRUSTLAYER_SAMPLINGSTEP,
            startval=dzcrustopt[i],
            maxjumpsize=DZ_CRUSTLAYER_MAXJUMP,
            nmaxsample=NB_SAMPLES)
        dzcrustlayers.append(dzcrustlayer)

    dzsedlayer = psmcsampling.Parameter(name='Thicknes of sediment layer',
                                        minval=dz_sediments_bounds[0],
                                        maxval=dz_sediments_bounds[1],
                                        step=DZ_SEDIMENTS_SAMPLINGSTEP,
                                        startval=dzsedopt,
                                        maxjumpsize=DZ_SEDIMENTS_MAXJUMP,
                                        nmaxsample=NB_SAMPLES)

    dzlayers = [dzsedlayer] + dzcrustlayers
    parameters = vscrustlayers + dzlayers

    # initial misfit and likelihood of parameters
    vsmodel_current = vsmodelinit
    misfit_current = vsmodel_current.misfit_to_vg(
        periods=PERIODS, vg=meanvg, sigmavg=sigmavg, storevg=True)  # storing vg model
    likelihood_current = np.exp(- misfit_current)

    # Monte Carlo sampling with Metropolis rule turned off (1st loop) and on (2nd loop)
    for switchon_Metropolis in [False, True]:
        s = "  Monte Carlo sampling of the {} distribution of the parameters"
        print s.format('prior' if not switchon_Metropolis else 'posterior')

        # (re-)initializing parameters
        for m in parameters:
            m.reinit()
        nrefused = 0
        vsmodels = []

        for isample in range(NB_SAMPLES):
            if switchon_Metropolis and (isample + 1) / 10 == (isample + 1) / float(10):
                s = '    Collected {} / {} samples ({:.1f} % of the moves accepted)'
                relaccepted = float(isample - nrefused) / isample
                print s.format(isample + 1, NB_SAMPLES, 100.0 * relaccepted)

            # adding sample to posterior distribution
            for m in parameters:
                m.addsample()
            vsmodels.append(vsmodel_current)

            # proposing next (random walk) move, which, if accepted, would
            # sample uniformly the parameters space: the strategy consists
            # in perturbating one Vs and one thickness selected at random
            # (and freezing all other parameters)
            freevslayer = np.random.choice(vscrustlayers)
            freedzlayer = np.random.choice(dzlayers)
            for vscrustlayer in vscrustlayers:
                vscrustlayer.frozen = False if vscrustlayer == freevslayer else True
            for dzlayer in dzlayers:
                dzlayer.frozen = False if dzlayer == freedzlayer else True
            for m in parameters:
                _ = m.propose_next()

            # we always accept the move if Metropolis rule is switched off
            if not switchon_Metropolis:
                for m in parameters:
                    m.accept_move()
                continue

            # we always refuse move towards an implausible model
            if not VS_MIN_INCREMENT is None:
                # checking Vs increments
                Vsincr = [m1.next() - m0.next()
                          for m0, m1 in zip(vscrustlayers[:-1], vscrustlayers[1:])]
                if any(dVs < VS_MIN_INCREMENT for dVs in Vsincr):
                    nrefused += 1
                    continue
            # checking Moho depth
            total_thickness = sum(m.next() for m in dzlayers)
            if not moho_depth_bounds[0] <= total_thickness <= moho_depth_bounds[1]:
                nrefused += 1
                continue

            # Vs model corresponding to the proposed parameters value
            vsmodel_next = psdepthmodel.VsModel(
                vs=np.r_[vs_sediments, [m.next() for m in vscrustlayers], vs_mantle],
                dz=[m.next() for m in dzlayers],
                ratio_vp_vs=ratio_vp_vs,
                ratio_rho_vs=ratio_rho_vs,
                name='Sample of posterior distribution')

            # the move is accepted with probability P = L_next / L_current,
            # with L = likelihood
            misfit_next = vsmodel_next.misfit_to_vg(periods=PERIODS,
                                                    vg=meanvg,
                                                    sigmavg=sigmavg,
                                                    storevg=True)  # storing vg model
            accept_move = psmcsampling.accept_move(misfit_current,
                                                   likelihood_current,
                                                   misfit_next)
            if not accept_move:
                nrefused += 1
                continue

            # move is accepted
            misfit_current = misfit_next
            likelihood_current = np.exp(- misfit_current)
            for m in parameters:
                m.accept_move()
            vsmodel_current = vsmodel_next

        if not switchon_Metropolis:
            # ==========================================
            # exporting histograms of prior distribution
            # ==========================================

            # out prefix = e.g., "1d models/Parana basin (prior distribution)"
            outprefix = os.path.join(DEPTHMODELS_DIR, locname + ' (prior distribution)')
            if usersuffix:
                outprefix += u'_{}'.format(usersuffix)

            outfile = u'{}.png'.format(outprefix)
            print "  exporting prior distributions to file: " + outfile
            fig = plt.figure(figsize=(5 * (nb_crust_layers + 1), 10), tight_layout=True)

            # Vs distributions
            for icol, m in enumerate(vscrustlayers, start=2):
                ax = fig.add_subplot(2, nb_crust_layers + 1, icol)
                m.hist(ax=ax, nburnt=NB_BURN)

            # thickness distributions
            for icol, m in enumerate(dzlayers, start=nb_crust_layers + 2):
                ax = fig.add_subplot(2, nb_crust_layers + 1, icol)
                m.hist(ax=ax, nburnt=NB_BURN)

            if os.path.exists(outfile):
                # backup
                shutil.copy(outfile, outfile + '~')
            fig.savefig(outfile, dpi=300)

            continue

    # ==============================================
    # dumping (1) observed vg arrays and (2) sampled
    # models from posterior distribution
    # ==============================================

    # out prefix = e.g., "1d models/Parana basin"
    outprefix = os.path.join(DEPTHMODELS_DIR, locname)
    if usersuffix:
        outprefix += u'_{}'.format(usersuffix)

    outfile = u'{}.pickle'.format(outprefix)
    print '    dumping observed vg and sampled models to file: ' + outfile
    if os.path.exists(outfile):
        # backup
        shutil.copy(outfile, outfile + '~')
    f = open(outfile, 'wb')
    # dumping observed group velocities
    pickle.dump(vgarrays, f, protocol=2)
    # dumping sampled models from posterior distribution
    pickle.dump(vsmodels, f, protocol=2)
    f.close()

    # ======================================================================
    # plotting results: 95% confidence intervals, posterior distributions...
    # =====================================================================

    outfile = u'{}.png'.format(outprefix)
    print "    plotting results and saving to file: " + outfile

    ncols = nb_crust_layers + 2
    nrows = 3

    quantiles = [2.5, 97.5]
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows), tight_layout=True)

    # 1st column, 2nd line: Vs versus z: inital model, representative model,
    # 95% interval and mean of posterior distribution
    ax = fig.add_subplot(nrows, ncols, ncols + 1)
    vsmodelinit.plot_model(ax=ax, color='b')  # initial model
    # 95% confidence interval
    vsz_arrays = [vsmodel.get_vs_at(DEPTHS) for vsmodel in vsmodels[NB_BURN:]]
    vs_1stquant, vs_2ndquant = np.percentile(vsz_arrays, quantiles, axis=0)
    ax.fill_betweenx(y=DEPTHS, x1=vs_1stquant, x2=vs_2ndquant, color='grey', alpha=0.3)
    # mean
    vsmean = np.mean(vsz_arrays, axis=0)
    ax.plot(vsmean, DEPTHS, '--', color='k')
    # representative model = model of posterior distribution closest to ensemble mean
    i = np.argmin([np.sum((vsz - vsmean)**2) for vsz in vsz_arrays])
    representative_model = vsmodels[NB_BURN:][i]
    representative_model.name = 'Representative model'
    representative_model.plot_model(ax=ax, color='r')
    # best-fitting model
    key = lambda vsmodel: vsmodel.misfit_to_vg(PERIODS, meanvg, sigmavg)
    best_model = min(vsmodels, key=key)
    best_model.name = 'Best-fitting model'
    best_model.plot_model(ax=ax, color='g')

    # 1st colum, 1st line: group velocities vs period: observed velocities,
    # initial model, representative model, 95% interval and mean
    # of posterior distribution
    ax = fig.add_subplot(nrows, ncols, 1)
    vsmodelinit.plot_vg(PERIODS, obsvgarrays=vgarrays, ax=ax, color='b')  # initial model
    representative_model.plot_vg(PERIODS, ax=ax, color='r')  # representative model
    best_model.plot_vg(PERIODS, ax=ax, color='g')  # best-fitting model
    # 95% confidence interval
    vgmodels = [vsmodel.stored_vg for vsmodel in vsmodels[NB_BURN:]]
    vg_1stquant, vg_2ndquant = np.percentile(vgmodels, quantiles, axis=0)
    ax.fill_between(x=PERIODS, y1=vg_2ndquant, y2=vg_1stquant, color='grey', alpha=0.3)
    # mean
    vgmean = np.mean(vgmodels, axis=0)
    ax.plot(PERIODS, vgmean, '--', color='k')
    ax.set_title(locname)

    # 1st line, next columns: posterior distribution of crustal layers' Vs
    for i, m in enumerate(vscrustlayers, start=3):
        ax = fig.add_subplot(nrows, ncols, i)
        m.hist(ax, nburnt=NB_BURN)

    # 2nd line, next columns: posterior distribution of
    # (sediment and crust) layers' thickness
    for i, m in enumerate(dzlayers, start=ncols + 2):
        ax = fig.add_subplot(nrows, ncols, i)
        m.hist(ax, nburnt=NB_BURN)

    # 3rd line, 1st column: misfit vs iteration nb
    misfits = [vsmodel.misfit_to_vg(PERIODS, vgarrays, sigmavg) for vsmodel in vsmodels]
    ax = fig.add_subplot(nrows, ncols, 2 * ncols + 1)
    ax.plot(misfits)
    ax.set_xlabel('Iteration nb')
    ax.set_ylabel('Misfit')
    ax.grid(True)

    # 3rd line, next columns: posterior distribution of depth of
    # base of crustal layers (including Moho depth)
    for icrustlayer in range(1, nb_crust_layers + 1):
        ax = fig.add_subplot(nrows, ncols, 2 * ncols + 1 + icrustlayer)
        zbase = dzsedlayer + sum(dzcrustlayers[:icrustlayer])
        if icrustlayer < nb_crust_layers:
            xlabel = 'Depth of base of crustal layer #{}'.format(icrustlayer)
        else:
            xlabel = 'Moho depth'
        zbase.hist(ax, nburnt=NB_BURN, xlabel=xlabel)

    if os.path.exists(outfile):
        # backup
        shutil.copy(outfile, outfile + '~')
    fig.savefig(outfile, dpi=300)
    fig.show()
