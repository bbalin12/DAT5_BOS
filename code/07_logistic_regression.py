# -*- coding: utf-8 -*-
"""
Logistic Regression.
Data Science Boston Feb 2015.
Logistic models of well switching in Bangladesh.
source: http://nbviewer.ipython.org/github/adparker/GADSLA_1403/blob/master/src/lesson07/Logistic%20Regression%20and%20Statsmodels%20tutorial.ipynb
"""

import numpy as np
import pandas
from statsmodels.formula.api import logit
from statsmodels.nonparametric import kde
import matplotlib.pyplot as plt
from patsy import dmatrix, dmatrices

df = pandas.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat', sep=' ', header=0, index_col=0)
print df.head()

model1 = logit('switch ~ I(dist/100.)', data = df).fit() # model1 is our fitted model.
print model1.summary()

def binary_jitter(x, jitter_amount = .05):
    '''
    Add jitter to a 0/1 vector of data for plotting.
    '''
    jitters = np.random.rand(*x.shape) * jitter_amount
    x_jittered = x + np.where(x == 1, -1, 1) * jitters
    return x_jittered

# First plot the Switch / No Switch dots vs distance to a safe well. Add jitter.
plt.plot(df['dist'], binary_jitter(df['switch'], .1), '.', alpha = .1)
# Now use the model to plot probability of switching vs distance (the green line).
sorted_dist = np.sort(df['dist'])
argsorted_dist = list(np.argsort(df['dist']))
predicted = model1.predict()[argsorted_dist]
plt.plot(sorted_dist, predicted, lw = 2)

kde_sw = kde.KDEUnivariate(df['dist'][df['switch'] == 1])
kde_nosw = kde.KDEUnivariate(df['dist'][df['switch'] == 0])

kde_sw.fit()
kde_nosw.fit()

plt.plot(kde_sw.support, kde_sw.density, label = 'Switch')
plt.plot(kde_nosw.support, kde_nosw.density, color = 'red', label = 'No Switch')
plt.xlabel('Distance (meters)')
plt.legend(loc = 'best')

model2 = logit('switch ~ I(dist / 100.) + arsenic', data = df).fit()
print model2.summary()

margeff =  model2.get_margeff(at = 'mean')
print margeff.summary()

logit_pars = model2.params
intercept = -logit_pars[0] / logit_pars[2]
slope = -logit_pars[1] / logit_pars[2]

dist_sw = df['dist'][df['switch'] == 1]
dist_nosw = df['dist'][df['switch'] == 0]
arsenic_sw = df['arsenic'][df['switch'] == 1]
arsenic_nosw = df['arsenic'][df['switch'] == 0]
plt.figure(figsize = (12, 8))
plt.plot(dist_sw, arsenic_sw, '.', mec = 'purple', mfc = 'None', 
         label = 'Switch')
plt.plot(dist_nosw, arsenic_nosw, '.', mec = 'orange', mfc = 'None', 
         label = 'No switch')
plt.plot(np.arange(0, 350, 1), intercept + slope * np.arange(0, 350, 1) / 100.,
         '-k', label = 'Separating line')
plt.ylim(0, 10)
plt.xlabel('Distance to safe well (meters)')
plt.ylabel('Arsenic level')
plt.legend(loc = 'best')

model3 = logit('switch ~ I(dist / 100.) + arsenic + I(dist / 100.):arsenic', 
                   data = df).fit()
print model3.summary()

model_form = ('switch ~ center(I(dist / 100.)) + center(arsenic) + ' +
              'center(I(educ / 4.)) + ' +
              'center(I(dist / 100.)) : center(arsenic) + ' + 
              'center(I(dist / 100.)) : center(I(educ / 4.)) + ' + 
              'center(arsenic) : center(I(educ / 4.))'
             )
model4 = logit(model_form, data = df).fit()
print model4.summary()

