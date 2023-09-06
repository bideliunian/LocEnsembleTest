# %%
import numpy as np
import pandas as pd
from plotnine import ggplot, geom_point, geom_line, aes


def get_df(model):
    saving_path = "D:/Angle/homo/Code/Distributions/Results/"
    methods = ['angle', 'ball', 'graph', 'energy']
    nreps = 500

    if model == 'model1':
        deltas = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    elif model == 'model2':
        deltas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
    elif model == 'model3':
        deltas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]

    pvalues = np.load(saving_path + model + "/pvalues"+ model + str(1) + ".npy")
    for i in range(1,101):
        mat = np.load(saving_path + model + "/pvalues"+ model + str(i) + ".npy")
        pvalues = np.concatenate((pvalues, mat), axis=1)

    power = np.sum(pvalues < 0.05, axis=1) / nreps
    df = pd.DataFrame(power, columns = methods)
    df['delta'] = deltas
    df = pd.melt(df, id_vars='delta', value_vars=methods)
    df.rename(columns = {'delta':'delta', 'variable':'method', 'value':'power'}, inplace = True)

    p = ggplot(df, aes(x='delta', y='power', color='method', group='method')) + \
        geom_point(aes(shape='method')) + geom_line(aes(linetype='method'), size=1) 

    return df, p
    

df1, p1 = get_df('model1')
df2, p2 = get_df('model2')
df3, p3 = get_df('model3')

# df1.to_csv('power_model1.csv', sep='\t')
# df2.to_csv('power_model2.csv', sep='\t')
# df3.to_csv('power_model3.csv', sep='\t')
# %%
