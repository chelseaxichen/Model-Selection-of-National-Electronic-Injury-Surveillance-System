import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython.display import display, HTML, Image

def bootstrap_sample(data, f, n=100):
    result = []
    for _ in range( n):
        sample = np.random.choice(data, len(data), replace=True)
        r = f(sample)
        result.append(r)
    return np.array(result)
  
#Helper function for 90% confidence interval provided posterior as input
def confidence_interval(posterior, ropeMin=False, ropeMax=False):
  lower, upper = stats.mstats.mquantiles(posterior, [0.05, 0.95])
  inf = float("inf")
  if not (ropeMin or ropeMax):
    return (lower, upper), None
  else:
    return (lower, upper),  np.mean(((ropeMin if ropeMin else -inf) <= posterior) & (posterior <= (ropeMax if ropeMax else inf)) )
	
def variable_details(series, disp=True):
  if series.dtype in ['int64', 'float64']:
    stat = series.describe()
    stat['Range'] = stat['max'] - stat['min']
    stat['IQR'] = stat['75%'] - stat['25%']
    stat['COV'] = 100* stat['std'] / stat['mean']
    stat['Variance'] = stat['std'] ** 2
    stat['Unique Values'] = series.nunique()
    stat['Skewed'] = "right [{:.2f}]".format(series.skew()) if series.skew() > 0 else "left [{:.2f}]".format(series.skew())
    display(HTML(stat.to_frame().T.to_html()))
  else:
    tmp = series.value_counts().to_frame().reset_index(drop=False).sort_index()
    cols = tmp.columns
    tmp = tmp.rename({cols[0]:cols[1], cols[1]: 'Count'}, axis=1)
    tmp['Frequency (%)'] = tmp.Count.div( tmp.Count.sum(axis=0)) * 100
    display(HTML(tmp.to_html(index=False))) if disp else ''
    return tmp

# function to display two plots: density histogram and box/swarm distribution plots
def single_variable_dist(series, rotation=0):
  if series.dtype in ['int64', 'float64']:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
    sns.distplot(series, hist=True, kde=False, ax=ax1)
    ax1.axvline(np.mean(series), color="DarkRed", label='mean')
    ax1.axvline(np.median(series), color="DarkOrange", label='median')
    ax1.legend(loc='best', frameon=False)
    ax1.set_ylabel('Frequency')
    ax1.xaxis.grid(False)

    sns.boxplot(x=series, width=0.5, boxprops=dict(alpha=.3), ax=ax2)
    fig.suptitle("{} Distribution".format(series.name), fontsize=14)
    ax2.set_ylabel('Values')
    ax2.xaxis.grid(False)
    plt.show()
  else:
    counts = series.value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.barplot(x=counts.index, y=counts, palette='colorblind', alpha=0.7, ax=ax)
    ax.set_title('{} Distribution'.format(series.name), fontsize=14)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Value')
    plt.xticks(rotation=rotation, ha='right')
    ax.xaxis.grid(False)
    plt.show()
	
def correlation(data, x, y):
    print("Correlation Coefficients:")
    pearsonr = stats.pearsonr(data[x], data[y])[0]
    spearmanr = stats.spearmanr(data[x], data[y])[0]
    print("r   =", pearsonr)
    print("rho =", spearmanr)
    return pearsonr, spearmanr

def correlation_matrix(df, method):
  corr = df.corr(method=method)
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  fig, ax = plt.subplots(figsize=(9,7))
  sns.heatmap(corr, mask=mask, cmap='Blues', square=True, annot=True, linewidths=0.5, ax=ax)
  plt.xticks(rotation=45, ha='right') 
  ax.set_title('Variable Correlations', fontsize=14)
  bottom, top = ax.get_ylim()
  ax.set_ylim(bottom + 0.5, top - 0.5)
  plt.show()

def describe_by_category(data, categorical, numeric):
    grouped = data.groupby(categorical)
    grouped_y = grouped[numeric].describe()
    return grouped_y
	
def multiboxplot(data, categorical, numeric, skip_data_points=True):
    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    grouped = data.groupby(categorical)
    labels = data[categorical].unique()
    labels = np.sort(labels)
    grouped_data = [grouped[numeric].get_group(k) for k in labels]
    patch = axes.boxplot(grouped_data, labels=labels, patch_artist=True, zorder=1)
    #eda.restyle_boxplot( patch)

    if not skip_data_points:
        for i, k in enumerate(labels):
            subdata = grouped[numeric].get_group( k)
            x = np.random.normal(i + 1, 0.01, size=len(subdata))
            axes.plot(x, subdata, 'o', alpha=0.4, color="DimGray", zorder=2)

    axes.set_xlabel(categorical)
    axes.set_ylabel(numeric)
    axes.set_title("Distribution of {0} by {1}".format(numeric, categorical))

    plt.show()
   
	
import statsmodels.api as sm
def lowess_scatter(data, x, y, jitter=0.0, axes=False, skip_lowess=False):

    if skip_lowess:
        fit = np.polyfit(data[x], data[y], 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 10)
        line = np.poly1d(fit)
        line_y = list(map(line, line_x))
    else:
        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)
        line_x = list(zip(*lowess))[0]
        line_y = list(zip(*lowess))[1]

    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    xs = data[x]
    if jitter > 0.0:
        xs = data[x] + stats.norm.rvs( 0, 0.5, data[x].size)

    axes.scatter(xs, data[y], marker="o", color="DimGray", alpha=0.5)
    axes.plot(line_x, line_y, color="DarkRed")

    title = "Plot of {0} v. {1}".format(x, y)
    if not skip_lowess:
        title += " with LOWESS"
    axes.set_title(title)
    axes.set_xlabel(x)
    axes.set_ylabel(y)

    plt.show()
    plt.close()