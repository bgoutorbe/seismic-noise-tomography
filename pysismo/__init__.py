import psconfig

# Not importing other modules here so that changing parameters in
# psconfig will take effect when importing other modules after.
# In other words, if for some reason you want to change some
# default parameters without touching the configuration file,
# you can do something like:
#
# >>> from pysismo import psconfig
# >>> psconfig.PERIOD_BANDS = [[5, 20], [10, 50]]
# >>> from pysismo import pscrosscorr
#
# In this example pscrosscorr will be initialized with the
# modified value of the parameter *PERIOD_BANDS*.

# import pscrosscorr, pserrors, psspectrum, psstation, psutils, pstomo