import matplotlib.pyplot as plt
import numpy as np

pb = np.load('hw10/data/pb.npy')
co = np.load('hw10/data/co.npy')
tt = np.load('hw10/data/tt.npy')
btb = np.load('hw10/data/btb.npy')

def plot_lj(xyz):
  # Compute distances between every pair of nodes
  D = np.zeros((7, 7))
  for i in range(7):
    for j in range(i):
      # Distance between i and j
      D[i][j] += (xyz[0][i] - xyz[0][j])**2
      D[i][j] += (xyz[1][i] - xyz[1][j])**2
      D[i][j] += (xyz[2][i] - xyz[2][j])**2
  D = np.sqrt(D)

  # For find smallest nonzero distance
  m = np.min(D[np.nonzero(D)])

  # Find elements of D within 2% of min
  L = np.abs(D - m) / m < 2 * 1e-2

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for i in range(7):
    for j in range(i):
      if L[i][j]:
        dx = [xyz[0][i], xyz[0][j]]
        dy = [xyz[1][i], xyz[1][j]]
        dz = [xyz[2][i], xyz[2][j]]
        ax.plot(dx, dy, dz, 'k')
  plt.show()

plot_lj(btb)
