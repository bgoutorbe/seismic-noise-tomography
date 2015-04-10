"""
Markov chain Monte-Carlo sampling of the parameter space
"""

import matplotlib.pyplot as plt
import numpy as np
np.random.seed()

EPS = 1E-6

# module variable containing samples from U(0, 1)
SIZE_SAMPLES_UNIFORM = int(1E6)
samples_uniform = np.random.uniform(size=SIZE_SAMPLES_UNIFORM)
isample_uniform = 0


class Parameter:
    """
    Class holding a model's parameter to be sampled uniformly
    along values from *minval* to *maxval* every *step*,
    using a Markov chain (random walk)
    """
    def __init__(self, name, minval, maxval, step, startval, maxjumpsize, nmaxsample,
                 frozen=False):
        """
        Initialization of parameter space, parameters of random walk
        and array of samples
        """
        self.name = name
        self.frozen = frozen  # if frozen, the value never changes

        # parameter space and random walk parameters
        self.values = np.arange(minval, maxval + step, step)
        self._nvalue = len(self.values)
        self._minval = minval
        self._maxval = maxval
        self._step = step
        self._maxjumpsize = maxjumpsize
        self._startval = startval

        self._neighborhoods = []
        for value in self.values:
            # neighborhood = all (different) values separated by up to *maxjumpsizes*
            neighboorhood = [i for i in range(self._nvalue)
                             if 0 < abs(value - self.values[i]) <= maxjumpsize]
            self._neighborhoods.append(neighboorhood)

        # parameter's current index
        i = np.argmin(np.abs(self.values - self._startval))
        if np.abs(self.values[i] - self._startval) > step:
            raise Exception('Starting value out of range')
        self._currentindex = i

        # initializing proposed next index
        self._proposednextindex = None

        # parameter's samples
        self.samples = np.zeros(nmaxsample)
        self.nsample = 0

    def reinit(self):
        """
        Reinitializes the parameter to its initial state
        """
        # parameter's current index back to start value
        self._currentindex = np.argmin(np.abs(self.values - self._startval))

        # reinitializing proposed next index
        self._proposednextindex = None

        # reinitializing parameter's samples
        self.samples[...] = 0.0
        self.nsample = 0

    def __repr__(self):
        s = '(ModelParameter)<{} randomly sampled between {}-{}>'
        return s.format(self.name, self._minval, self._maxval)

    def __add__(self, other):
        """
        Adds two parameter
        @type other: Parameter
        """

        if other == 0:
            # 0 + self = self (to allow sum([parameter1, parameter2...])
            return self

        if abs(self._step - other._step) > EPS:
            s = "Warning: parameters {} and {} have different sampling steps"
            print s.format(self, other)
        if abs(self._maxjumpsize - other._maxjumpsize) > EPS:
            s = "Warning: parameters {} and {} have different max jump size"
            print s.format(self, other)
        if self.nsample != other.nsample:
            raise Exception("Parameters must have the same nb of samples")

        m = Parameter(name=u"{} + {}".format(self.name, other.name),
                      minval=self._minval + other._minval,
                      maxval=self._maxval + other._maxval,
                      step=max(self._step, other._step),
                      maxjumpsize=max(self._maxjumpsize, other._maxjumpsize),
                      startval=self.current() + other.current(),
                      nmaxsample=max(np.size(self.samples), np.size(other.samples)))

        # filling existing samples
        m.nsample = self.nsample
        m.samples[:m.nsample] = self.samples[:m.nsample] + other.samples[:m.nsample]

        return m

    def __radd__(self, other):
        return self + other

    def current(self):
        """
        Current value
        """
        return self.values[self._currentindex]

    def next(self):
        """
        Next proposed value
        """
        if self._proposednextindex is None:
            raise Exception("No next value proposed yet.")
        return self.values[self._proposednextindex]

    def propose_next(self):
        """
        Proposing next value, using a random walk that samples
        uniformly the parameter space
        """
        if self._nvalue > 1 and not self.frozen:
            self._proposednextindex = random_walk_nextindex(
                self._currentindex, self._nvalue, neighborhoods=self._neighborhoods)
        else:
            self._proposednextindex = self._currentindex
        return self.values[self._proposednextindex]

    def accept_move(self):
        """
        Moving to proposed next value
        """
        if self._proposednextindex is None:
            raise Exception("No next value proposed yet.")

        self._currentindex = self._proposednextindex
        self._proposednextindex = None

    def addsample(self):
        """
        Adding current parameter value to samples
        """
        if self.nsample >= len(self.samples):
            raise Exception("Max number of samples reached")

        self.samples[self.nsample] = self.current()
        self.nsample += 1

    def hist(self, ax=None, nburnt=0, xlabel=None):
        """
        Plotting histogram of samples value or samples increment
        """
        # creating figure if not given as input
        fig = None
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # histogram of value of samples
        bins = np.arange(self._minval - 0.5 * self._step,
                         self._maxval + 1.5 * self._step,
                         self._step)
        samples = self.samples[nburnt:self.nsample]
        ax.hist(samples, bins=bins, normed=True, label='sampled distribution')

        # prior (uniform) distribution
        if self._maxval > self._minval:
            x = 2 * [self._minval] + 2 * [self._maxval]
            y = [0.0] + 2 * [1.0 / (self._maxval - self._minval)] + [0.0]
            ax.plot(x, y, '-', lw=2, color='grey', label='prior distribution')

        # legend, labels and title
        ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
        ax.set_xlabel(self.name if not xlabel else xlabel)
        ax.set_ylabel('Probability density')
        ax.set_title('Nb of samples: {}'.format(len(samples)))
        ax.grid(True)

        # statistics
        s = "Mean & std dev:\n{:.3G} +/- {:.3G}".format(np.mean(samples),
                                                        np.std(samples))
        quantiles = np.percentile(samples, [2.5, 97.5])
        s += "\n95% confidence interval:\n{:.3G}, {:.3G}".format(*quantiles)
        ax.text(min(ax.get_xlim()), max(ax.get_ylim()), s,
                fontdict={'fontsize': 10},
                horizontalalignment='left',
                verticalalignment='top',
                bbox={'color': 'w', 'alpha': 0.8})

        if fig:
            fig.show()

    def plot(self):
        plt.plot(self.samples[:self.nsample], '-')
        plt.xlabel('Sample number')
        plt.ylabel(self.name)
        plt.show()


def accept_move(misfit_current, likelihood_current, misfit_proposednext):
    """
    Is move accepted? Yes with probability P = L_next / L_current,
    with L = likelihood
    @rtype: bool
    """
    if misfit_proposednext <= misfit_current:
        return True

    # gaussian likelihood
    P = np.exp(-misfit_proposednext) / likelihood_current
    return True if sample_uniform() < P else False


def random_walk_nextindex(currentindex, npoints, maxstepsize=1, neighborhoods=None):
    """
    Next index of a random walk that samples uniformly all
    indexes between 0 and npoints - 1.

    Neighbors are either points sperated by up to *maxstepsize*
    from current index, or are given by *neighborhoods[currentindex]*.

    Next index is chosen equiprobably amongst neighbors.
    If the proposed next index has less (or equal) neighbors than
    the current index, the move is always accepted. Else, the
    move is accepted with probability:
    P = n_neighbors_current / n_neighbors_next

    @type currentindex: int
    @type npoints: int
    @type maxstepsize: int
    @type neighborhoods: list of (list of int)
    @rtype: int
    """
    # neighborhood of current point...
    if neighborhoods:
        # ... given as input
        ineighbours = neighborhoods[currentindex]
    else:
        # ... all points of the grid separated by up to
        # *maxstepsize* from current point
        ineighbours = neighborhood(currentindex, npoints, maxdist=maxstepsize)

    # proposed move, chosen equiprobably amongst neighbours
    u = sample_uniform()
    nextindex = ineighbours[int(u * len(ineighbours))]

    # nb of neighbours of proposed point
    if neighborhoods:
        nnextneighbours = len(neighborhoods[nextindex])
    else:
        dist2edge = min(nextindex, npoints - 1 - nextindex)
        nnextneighbours = maxstepsize + min(maxstepsize, dist2edge)

    # the move is accepted with probability
    # P = min(1, nb current neighbours / nb next neighbours)
    P = float(len(ineighbours)) / float(nnextneighbours)
    return nextindex if (P >= 1 or sample_uniform() < P) else currentindex


def random_walk(start, grid, nstep=np.Infinity, maxstepsize=1, likelihood=None):
    """
    [Metropolis] random walk with jumps of up to *maxstepsize* that:

    - samples uniformly all values of *grid* if no *likelihood*
      function is given (all moves are accepted)

    - accepts the moves with probability L_new / L_current,
      (where L is the likelihood) if a *likelihood* function is
      given, thus sampling k.U(x).L(x)

    Returns an interator of length *nstep*, or an infinite
    iterator if nstep = infinity.

    @type start: float
    @type grid: L{numpy.ndarray}
    @type nstep: int
    @type maxstepsize: int
    @type likelihood: function
    """
    if not min(grid) <= start <= max(grid):
        raise Exception("Starting point not within grid limits")

    # preparing list of neighborhoods
    neighborhoods = []
    npoints = len(grid)
    for i in range(npoints):
        neighborhoods.append(neighborhood(i, npoints, maxdist=maxstepsize))

    # starting index and step nb
    currentindex = np.abs(grid - start).argmin()
    for _ in range(nstep):
        # yielding current value
        yield grid[currentindex]

        # likelihood of current point (1 if no likelihood func given)
        L_current = likelihood(grid[currentindex]) if likelihood else 1.0

        # proposed ove
        nextindex = random_walk_nextindex(currentindex, npoints,
                                          maxstepsize=maxstepsize,
                                          neighborhoods=neighborhoods)

        # probability to accept move (always accepted is no likelihood func given)
        L_new = likelihood(grid[nextindex]) if likelihood else L_current + 1.0
        P = L_new / L_current
        currentindex = nextindex if P >= 1.0 or sample_uniform() <= P else currentindex


def sample_uniform():
    """
    Returns a single sample of the uniform distribution
    U(0, 1), from samples drawn and stored in a global
    variable.
    """
    global samples_uniform, isample_uniform

    # sample of U(0, 1)
    u = samples_uniform[isample_uniform]

    # moving to next index of samples global array
    isample_uniform += 1
    if isample_uniform >= len(samples_uniform):
        # exhausted all samples -> re-drawing samples from U(0, 1)
        samples_uniform = np.random.uniform(size=SIZE_SAMPLES_UNIFORM)
        isample_uniform = 0

    return u


def neighborhood(index, npoints, maxdist=1):
    """
    Returns the neighbourhood of the current index,
    = all points of the grid separated by up to
    *maxdist* from current point.

    @type index: int
    @type npoints: int
    @type maxdist int
    @rtype: list of int
    """
    return [index + i for i in range(-maxdist, maxdist + 1)
            if i != 0 and 0 <= index + i <= npoints - 1]