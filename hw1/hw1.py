import random

# Tent map on the interval [0, 1]
def f(x):
  return (2 * x) if (x < 0.5) else 2 - (2 * x)

def iterate(n):
  x = random.random()
  print(x)
  for i in range(n):
    x = f(x)
    print(x)

iterate(55)
