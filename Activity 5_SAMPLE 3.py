import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=50)
test = np.linspace(0, 2)
likelihood_out = sts.norm.pdf(1.7, mu, scale=0.1)
uniform_dist = sts.uniform.pdf(mu)

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()
