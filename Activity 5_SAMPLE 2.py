import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

mu = np.linspace(1.65, 1.8, num=50)
likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title(r"likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability density/Likelihood")
plt.xlabel(r"Value of $\mu$")
plt.show()
