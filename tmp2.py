"""
hexbin is an axes method or pyplot function that is essentially
a pcolor of a 2-D histogram with hexagonal cells.  It can be
much more informative than a scatter plot; in the first subplot
below, try substituting 'scatter' for 'hexbin'.
"""

fig = plt.hist2d(data_a[:,0], data_a[:,1], bins=500)
plt.axis([-1, 1, -1, 1])
cb = plt.colorbar()
cb.set_label('counts')
plt.show()