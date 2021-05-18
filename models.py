import numpy as np
import scipy.stats as stats
import patsy
import sklearn.linear_model as linear
import random
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


# we're not currently using this because the LaTeX
# experience is so awful (it doesn't use HTML but
# the plaintext representation).
from IPython.display import HTML, display_html
from tabulate import tabulate

ALGORITHMS = {
    "linear": linear.LinearRegression,
    "ridge": linear.Ridge,
    "lasso": linear.Lasso
}

def summarize(formula, X, y, model, style='linear'):
    result = {}
    result["formula"] = formula
    result["n"] = len(y)
    result["model"] = model
    # I think this is a bug in Scikit Learn 
    # because lasso should work with multiple targets.
    if style == "lasso":
        result["coefficients"] = model.coef_
    else:
        result["coefficients"] =  model.coef_[0]
    result["r_squared"] = model.score(X, y)
    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"]  = y
    sum_squared_error = sum([e**2 for e in result["residuals"]])[0]

    n = len(result["residuals"])
    k = len(result["coefficients"])
    
    result["sigma"] = np.sqrt(sum_squared_error / (n - k))
    return result

def linear_regression(formula, data=None, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    params["fit_intercept"] = False

    y, X = patsy.dmatrices(formula, data, return_type= "matrix")
    algorithm = ALGORITHMS[style]
    algo = algorithm(**params)
    model = algo.fit( X, y)

    result = summarize(formula, X, y, model, style)
   
    return result

def logistic(z):
    return 1.0 / (1.0 + np.exp( -z))

def logistic_regression(formula, data=None):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    result = {}
    result["formula"] = formula
    result["n"] = data.shape[ 0]

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    y = np.ravel( y) # not sure why this is needed for LogisticRegression but not LinearRegression

    model = linear.LogisticRegression(fit_intercept=False).fit(X, y)
    result["model"] = model

    result["coefficients"] = model.coef_[ 0]

    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat 
    result["y"] = y

    # efron's pseudo R^2
    y_bar = np.mean(y)
    pr = model.predict_proba(X).transpose()[1]
    result["probabilities"] = pr
    efrons_numerator = np.sum((y - pr)**2) 
    efrons_denominator = np.sum((y-y_bar)**2)
    result["r_squared"] = 1 - (efrons_numerator/efrons_denominator)

    # error rate
    result["error"] = np.sum(np.abs(result["residuals"]))/result["n"]*100

    n = len(result["residuals"])
    k = len(result["coefficients"])

    return result

def bootstrap_linear_regression(formula, data=None, samples=100, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    
    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split( "+")]
    variables = ["intercept"] + variables
    bootstrap_results[ "variables"] = variables
    
    coeffs = []
    sigmas = []
    rs = []

    n = data.shape[0]
    bootstrap_results['n'] = n
    
    for i in range( samples):
        sampling_indices = [i for i in [np.random.randint(0, n - 1) for _ in range(0, n)]]
        sampling = data.loc[sampling_indices]
        
        results = linear_regression(formula, data=sampling, style=style, params=params)
        coeffs.append( results["coefficients"])
        sigmas.append( results["sigma"])
        rs.append( results["r_squared"])
    
    coeffs = pd.DataFrame(coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name="sigma")
    rs = pd.Series(rs, name="r_squared")

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_sigma"] = sigmas
    bootstrap_results["resampled_r^2"] = rs
    
    result = linear_regression(formula, data=data)
    
    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["sigma"] = result[ "sigma"]
    bootstrap_results["r_squared"] = result[ "r_squared"]
    bootstrap_results["model"] = result["model"]
    bootstrap_results["y"] = result["y"]
    bootstrap_results["y_hat"] = result["y_hat"]
    return bootstrap_results

def bootstrap_logistic_regression(formula, data=None, samples=100):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    
    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split("+")]
    variables = ["intercept"] + variables
    bootstrap_results["variables"] = variables
    
    coeffs = []
    sigmas = []
    rs = []

    n = data.shape[0]
    bootstrap_results['n'] = n
    
    for i in range(samples):
        sampling_indices = [i for i in [np.random.randint(0, n - 1) for _ in range( 0, n)]]
        sampling = data.loc[sampling_indices]
        
        results = logistic_regression(formula, data=sampling)
        coeffs.append(results['coefficients'])
        sigmas.append(results['error'])
        rs.append(results['r_squared'])
    
    coeffs = pd.DataFrame (coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name='error')
    rs = pd.Series(rs, name='r_squared')

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_error"] = sigmas
    bootstrap_results["resampled_r^2"] = rs
    
    result = logistic_regression(formula, data=data)
    
    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["error"] = result[ "error"]
    bootstrap_results["r_squared"] = result["r_squared"]
    bootstrap_results["model"] = result["model"]
    return bootstrap_results

def fmt(n, sd=2):
    return (r"{0:." + str(sd) + "f}").format(n)

def results_table(fit, sd=2,bootstrap=False, is_logistic=False):
    result = {} 
    result["model"] = [fit["formula"]]

    variables = [""] + fit["formula"].split("~")[1].split( "+")
    coefficients = [] 

    if bootstrap:
        bounds = fit["resampled_coefficients"].quantile([0.025, 0.975])
        bounds = bounds.transpose()
        bounds = bounds.values.tolist()
        for i, b in enumerate(zip(variables, fit["coefficients"], bounds)):
            coefficient = [b[0], r"$\beta_{0}$".format(i), fmt(b[1], sd), fmt(b[2][0], sd), fmt(b[2][1], sd)]
            if is_logistic:
                if i == 0:
                    coefficient.append(fmt(logistic(b[1]), sd))
                else:
                    coefficient.append(fmt(b[1]/4, sd))
            coefficients.append(coefficient)
    else:
        for i, b in enumerate(zip(variables, fit["coefficients"])):
            coefficients.append([b[0], r"$\beta_{0}$".format(i), fmt(b[1], sd)])
    result["coefficients"] = coefficients

    error = r"$\sigma$"
    r_label = r"$R^2$"
    if is_logistic:
        error = "Error ($\%$)"
        r_label = r"Efron's $R^2$"
    if bootstrap:
        sigma_bounds = stats.mstats.mquantiles(fit["resampled_sigma"], [0.025, 0.975])
        r_bounds = stats.mstats.mquantiles(fit["resampled_r^2"], [0.025, 0.975])
        metrics = [
            [error, fmt(fit["sigma"], sd), fmt(sigma_bounds[0], sd), fmt(sigma_bounds[1], sd)], 
            [r_label, fmt(fit["r_squared"], sd), fmt(r_bounds[0], sd), fmt(r_bounds[1], sd)]]
    else:
        metrics = [
            [error, fmt(fit["sigma"], sd)], 
            [r_label, fmt(fit["r_squared"], sd)]]

    result["metrics"] = metrics

    # this is a kludge until I can figure out a better way
    # for the LaTeX version to use HTML instead of plain text
    # as the fall back.
    temp_result = f"Model: {result['model'][0]}\n"
    rows = []
    if bootstrap:
        rows.append(["", "", "", "95% BCI"])
    if is_logistic:
        if bootstrap:
            header = ["Cofficients", "", "Mean", "Lo", "Hi", "P(y=1)"]
        else:
            header = ["Coefficients", "", "Value"]
    else:
        if bootstrap:
            header = ["Coefficients", "", "Mean", "Lo", "Hi"]
        else:
            header = ["Coefficients", "", "Value"]
    rows.append(header)
    for row in result["coefficients"]:
        rows.append(row)
    rows.append([])
    if bootstrap:
        rows.append(["Metrics", "Mean", "Lo", "Hi"])
    else:
        rows.append(["Metrics", "Value"])
    for row in result["metrics"]:
        rows.append(row)
    temp_result += tabulate(rows)
    return temp_result

class ResultsView(object):
    def __init__(self, content, bootstrap=False, is_logistic=False):
        self.content = content
        self.bootstrap = bootstrap
        self.is_logistic = is_logistic

    def _repr_html_(self):
        span = "2"
        if self.bootstrap and not self.is_logistic:
            span = "3"
        if self.bootstrap and self.is_logistic:
            span = "5"
        result = r"<table><tr><th colspan=" + span + r">Linear Regression Results</th></tr>"
        if self.is_logistic:
            result = r"<table><tr><th colspan=" + span + r">Logistic Regression Results</th></tr>"
        result += r"<th colspan=" + span + r">Coefficients</th></tr>"
        coefficients = self.content["coefficients"]
        template = r""
        headers = r""
        if self.is_logistic:
            if self.bootstrap:
                header = r"<tr><th>$\theta$</th><th></th><th>95% BCI</th><th>P(y=1)</th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td><td>({3}, {4})</td><td>{5}</td></tr>"
            else:
                header = r"<tr><th>$\theta$</th><th></th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td></tr>"
        else:
            if self.bootstrap:
                header = r"<tr><th>$\theta$</th><th></th><th>95% BCI</th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td><td>({3}, {4})</td></tr>"
            else:
                header = r"<tr><th>$\theta$</th><th></th></tr>"
                template = r"<tr><td>{0} ({1})</td><td>{2}</td></tr>"
        result += header
        for coefficient in coefficients:
            result += template.format(*coefficient)
        
        result += r"<tr><th colspan=" + span + ">Metrics</th></tr>"

        metrics = self.content["metrics"]
        template = r"<tr><td>{0}</td><td>{1}</td></tr>"
        if self.bootstrap:
            template = r"<tr><td>{0}</td><td>{1}</td><td>({2}, {3})</td><td></td></tr>"

        for metric in metrics:
            result += template.format(*metric)
        
        result += r"</table>"
        return result

    def x__repr_latex_(self):
        span = 2
        if self.bootstrap and not self.is_logistic:
            span = 3 
        if self.bootstrap and self.is_logistic:
            span = 4
        result = r"\begin{table}[!htbp] \begin{tabular}{" + (r"l" * span) + r"} \hline \multicolumn{" + str(span) + r"}{c}{\textbf{Linear Regression}} \\ \hline \hline "
        if self.is_logistic:
            result = r"\begin{table}[!htbp] \begin{tabular}{"+ (r"l" * span) + r"} \hline \multicolumn{" + str(span) + r"}{c}{\textbf{Logistic Regression}} \\ \hline \hline "

        result += r"\multicolumn{" + str(span) + r"}{l}{\textbf{Coefficients}}        \\ \hline "
        coefficients = self.content["coefficients"]
        template = r""
        headers = r""
        if self.is_logistic:
            if self.bootstrap:
                header = r"$\theta$       &          & 95\% BCI     & P(y=1)\\"
                template = r"{0} ({1})      & {2}   & ({3}, {4})   & {5}  \\"
            else:
                header = r"$\theta$                  &                    \\"
                template = r"{0} ({1})                & {2}               \\"
        else:
            if self.bootstrap:
                header = r"$\theta$       &          & 95\% BCI           \\"
                template = r"{0} ({1})      & {2}   & ({3}, {4})          \\"
            else:
                header = r"$\theta$                  &                    \\"
                template = r"{0} ({1})                & {2}               \\"
        result += header
        for coefficient in coefficients:
            coefficient[0] = coefficient[0].replace('_', '\_')
            result += template.format(*coefficient)
        
        result += r"\hline \multicolumn{" + str(span) + r"}{l}{\textbf{Metrics}}             \\ \hline "

        metrics = self.content["metrics"]
        template = r"{0}                & {1}               \\"
        if self.bootstrap:
            template = r"{0}      & {1}   & ({2}, {3})          \\"

        for metric in metrics:
            result += template.format(*metric)
        result += r"\hline"
        result += r"\end{tabular}\end{table}"
        return result

def print_csv(table):
    print("Linear Regression")
    print("Coefficients")
    for item in table["coefficients"]:
        print(','.join(item))
    print("Metrics")
    for item in table["metrics"]:
        print(','.join(item))
    
def simple_describe_lr(fit):
    results = r"""
    <table>
    <tr><th> model </th><td> {formula} </td></tr>
    <tr><th colspan=2>coefficients</th></tr>
    """.format(**fit)

    results += r"<tr><th> $\beta_0$ </th><td> {0} </td></tr>".format(fit["coefficients"][0])

    variables = fit["formula"].split("~")[1].split("+")
    for i, b in enumerate(zip(variables, fit["coefficients"][1:])):
        results += r"<tr><td> {1} ($\beta_{0}$) </td><td> {2} </td></tr>".format(i + 1, b[0], b[1])

    results += r"""
    <tr><th colspan=2>metrics</th></tr>
    <tr><td> $\sigma$ </td><td> {sigma} </td></tr>
    <tr><td> $R^2$ </td><td> {r_squared} </td></tr>
    </table>
    """.format(**fit)
    return HTML(results)

def simple_describe_lgr(fit):
    results = r"""
	<table>
	<tr><th> model </th><td> {formula} </td></tr>
	<tr><th colspan=2>coefficients</th></tr>""".format(**fit)

    results += r"<tr><th> $\beta_0$ </th><td> {0} </td></tr>".format(fit["coefficients"][0])

    variables = fit["formula"].split("~")[1].split( "+")
    for i, b in enumerate(zip(variables, fit["coefficients"][1:])):
        results += r"<tr><td> {1} ($\beta_{0}$) </td><td> {2} </td></tr>".format(i + 1, b[0], b[1])

    results += r"""
    <tr><th colspan=2>metrics</th></tr>
    <tr><td> Error (%) </td><td> {error} </td></tr>
    <tr><td> $R^2$ </td><td> {r_squared} </td></tr>
    </table>
	""".format(**fit)
    return HTML(results)

def describe_bootstrap_lr(fit, significant_digits=2):
    sd = significant_digits
    results = r"""
    <table>
    <tr><th> model </th><td> {formula} </td></tr>
    <tr><th colspan=2>coefficients</th><th>95% BCI</th></tr>
    """.format(**fit)

    bounds = fit["resampled_coefficients"].quantile([0.025, 0.975])
    bounds = bounds.transpose()
    bounds = bounds.values.tolist()

    results += (r"<tr><th> $\beta_0$ </th><td> {0:." + 
        str(sd) + r"f} </td><td>({1:." +
        str(sd) + r"f}, {2:." + str(sd) + "f})</td></tr>").format(fit["coefficients"][0], bounds[0][0], bounds[0][1])

    variables = fit["formula"].split("~")[1].split( "+")
    for i, b in enumerate(zip(variables, fit["coefficients"][1:], bounds[1:])):
        results += (r"<tr><td> {1} ($\beta_{0}$) </td><td> {2:." +
            str(sd) + r"f} </td><td>({3:." + 
            str(sd) + r"f}, {4:." +
            str(sd) + r"f})</tr>").format(i + 1, b[0], b[1], b[2][0], b[2][1])

    sigma_bounds = stats.mstats.mquantiles(fit["resampled_sigma"], [0.025, 0.975])
    r_bounds = stats.mstats.mquantiles(fit["resampled_r^2"], [0.025, 0.975])

    results += r"<tr><th colspan=2>metrics</th><th>95% BCI</th></tr>"
    results += (r"<tr><th> $\sigma$ </th><td> {0:." +
         str(sd) + r"f} </td><td>({1:." +
         str(sd) + r"f}, {2:." +
         str(sd) + r"f}) </td></tr>").format(fit["sigma"], sigma_bounds[0], sigma_bounds[1])
    results += (r"<tr><th> $R^2$ </th><td> {0:." +
         str(sd) + r"f} </td><td>({1:." + 
         str(sd) + r"f}, {2:." +
         str(sd) + r"f})</td></tr>").format(fit["r_squared"], r_bounds[0], r_bounds[1])
    results += r"</table>"
    return HTML(results)

def describe_bootstrap_lgr(fit, significant_digits=2):
    sd = significant_digits
    results = r"""
    <table>
    <tr><th> model </th><td> {formula} </td></tr>
    <tr><th colspan=2>coefficients</th><th>95% BCI</th><th>P(y=1)</th></tr>
    """.format(**fit)

    bounds = fit["resampled_coefficients"].quantile([0.025, 0.975])
    bounds = bounds.transpose()
    bounds = bounds.values.tolist()

    results += (r"<tr><th> $\beta_0$ </th><td> {0:." + 
        str(sd) + r"f} </td><td>({1:." +
        str(sd) + r"f}, {2:." + 
        str(sd) + r"f})</td><td>{3:." +
        str(sd) + r"f}</td></tr>"
        ).format(fit["coefficients"][0], bounds[0][0], bounds[0][1], logistic(fit["coefficients"][0]))

    variables = fit["formula"].split("~")[1].split( "+")
    for i, b in enumerate(zip(variables, fit["coefficients"][1:], bounds[1:])):
        results += (r"<tr><td> {1} ($\beta_{0}$) </td><td> {2:." +
            str(sd) + r"f} </td><td>({3:." + 
            str(sd) + r"f}, {4:." +
            str(sd) + r"f})</td><td>{5:." +
            str(sd) + r"f}</td></tr>").format(i + 1, b[0], b[1], b[2][0], b[2][1], b[1]/4)

    sigma_bounds = stats.mstats.mquantiles(fit["resampled_error"], [0.025, 0.975])
    r_bounds = stats.mstats.mquantiles(fit["resampled_r^2"], [0.025, 0.975])

    results += r"<tr><th colspan=2>metrics</th><th>95% BCI</th></tr>"
    results += (r"<tr><td> Error (%) </td><td> {0:." +
         str(sd) + r"f} </td><td>({1:." +
         str(sd) + r"f}, {2:." +
         str(sd) + r"f}) </td></tr>").format(fit["error"], sigma_bounds[0], sigma_bounds[1])
    results += (r"<tr><th> $R^2$ </th><td> {0:." +
         str(sd) + r"f} </td><td>({1:." + 
         str(sd) + r"f}, {2:." +
         str(sd) + r"f})</td></tr>").format(fit["r_squared"], r_bounds[0], r_bounds[1])
    results += r"</table>"
    return HTML(results)

def strength(pr):
    if 0 <= pr <= 0.33:
        return "weak"
    if 0.33 < pr <= 0.66:
        return "mixed"
    return "strong"

# {"var1": "+", "var2": "-"}
def evaluate_coefficient_predictions(predictions, result):
    coefficients = result["resampled_coefficients"].columns
    for coefficient in coefficients:
        if coefficient == 'intercept':
            continue
        if predictions[coefficient] == '+':
            pr = np.mean(result["resampled_coefficients"][coefficient] > 0)
            print("{0} P(>0)={1:.3f} ({2})".format(coefficient, pr, strength(pr)))
        else:
            pr = np.mean(result["resampled_coefficients"][coefficient] < 0)
            print("{0} P(<0)={1:.3f} ({2})".format(coefficient, pr, strength(pr)))

def adjusted_r_squared(result):
    adjustment = (result["n"] - 1)/(result["n"] - len(result["coefficients"]) - 1 - 1)
    return 1 - (1 - result["r_squared"]) * adjustment

def plot_residuals(result, variables, data):
    figure = plt.figure(figsize=(19,9))
    plots = len(variables)
    rows = (plots // 3) + 1
    residuals = np.array([r[0] for r in result["residuals"]])
    limits = max( np.abs(residuals.min()), residuals.max())
    
    n = result["n"]
    for i, variable in enumerate(variables):
        axes = figure.add_subplot(rows, 3, i + 1)

        keyed_values = sorted(zip(data[variable].values, residuals), key=lambda x: x[ 0])
        ordered_residuals = [x[1] for x in keyed_values]

        axes.plot(list(range(0, n)), ordered_residuals, '.', color="dimgray", alpha=0.75)
        axes.axhline(y=0.0, xmin=0, xmax=n, c="firebrick", lw=2, alpha=0.5)
        axes.set_ylim((-limits, limits))
        axes.set_ylabel("residuals")
        axes.set_xlabel(variable, fontsize=14)

    plt.tight_layout()
    plt.show()
    return residuals
	
def boostrap_prediction(result, feature_vals):
  results = []
  for coeffs in result["resampled_coefficients"].itertuples():
    est=0
    for feature in feature_vals.keys():
      est += feature_vals[feature] * getattr(coeffs, feature)
    results.append( est)
  return np.array(results)

def chunk(xs, n):
    k, m = divmod(len(xs), n)
    return [xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
	
def linear_cross_validation(algorithm, formula, data, evaluate, fold_count=10, repetitions=3):
    indices = list(range(len(data)))
    metrics = []
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            result = algorithm(formula, data=train_data)
            model = result["model"]
            y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
            # y = np.ravel(y) # might need for logistic regression
            results = summarize(formula, X, y, model)
            metric = evaluate(results)
            metrics.append(metric)
    return metrics
	
def logistic_cross_validation(algorithm, formula, data, evaluate, fold_count=10, repetitions=3):
    indices = list(range(len(data)))
    metrics = []
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            result = algorithm(formula, data=train_data)
            model = result["model"]
            y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
            y = np.ravel(y) 
            results = summarize(formula, X, y, model)
            metric = evaluate(results)
            metrics.append(metric)
    return metrics
	
def data_collection():
    result = dict()
    result[ "train"] = defaultdict(list)
    result[ "test"] = defaultdict(list)
    return result
	
def learning_curves(algorithm, formula, data, evaluate, fold_count=10, repetitions=3, increment=1):
  indices = list(range(len(data)))
  results = data_collection()
    
  for _ in range(repetitions):
      random.shuffle(indices)
      folds = chunk(indices, fold_count)
      for fold in folds:
          test_data = data.iloc[fold]
          train_indices = [idx for idx in indices if idx not in fold]
          train_data = data.iloc[train_indices]
          for i in list(range(increment, 100, increment)) + [100]: # ensures 100% is always picked.
              # the indices are already shuffled so we only need to take ever increasing chunks
              train_chunk_size = int(np.ceil((i/100)*len(train_indices)))
              train_data_chunk = data.iloc[train_indices[0:train_chunk_size]]
              # we calculate the model
              result = algorithm(formula, data=train_data_chunk)
              model = result["model"]
              # we calculate the results for the training data subset
              y, X = patsy.dmatrices(formula, train_data_chunk, return_type="matrix")
              result = summarize(formula, X, y, model)
              metric = evaluate(result)
              results["train"][i].append( metric)
              
              # we calculate the results for the test data.
              y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
              result = summarize(formula, X, y, model)
              metric = evaluate(result)
              results["test"][i].append(metric)
  # process results
  # Rely on the CLT...
  statistics = {}
  for k, v in results["train"].items():
      statistics[k] = (np.mean(v), np.std(v))
  results["train"] = statistics
  statistics = {}
  for k, v in results["test"].items():
      statistics[k] = (np.mean(v), np.std(v))
  results["test"] = statistics
  return results
	
def results_to_curves(curve, results):
    all_statistics = results[curve]
    keys = list(all_statistics.keys())
    keys.sort()
    mean = []
    upper = []
    lower = []
    for k in keys:
        m, s = all_statistics[k]
        mean.append(m)
        upper.append(m + 2 * s)
        lower.append(m - 2 * s)
    return keys, lower, mean, upper
	
def plot_learning_curves(results, metric, zoom=False):
    figure = plt.figure(figsize=(10,6))

    axes = figure.add_subplot(1, 1, 1)

    xs, train_lower, train_mean, train_upper = results_to_curves("train", results)
    _, test_lower, test_mean, test_upper = results_to_curves("test", results)

    axes.plot(xs, train_mean, color="steelblue")
    axes.fill_between( xs, train_upper, train_lower, color="steelblue", alpha=0.25, label="train")
    axes.plot(xs, test_mean, color="firebrick")
    axes.fill_between(xs, test_upper, test_lower, color="firebrick", alpha=0.25, label="test")
    axes.legend()
    axes.set_xlabel("training set (%)")
    axes.set_ylabel(metric)
    axes.set_title("Learning Curves")

    if zoom:
        y_lower = int(0.9 * np.amin([train_lower[-1], test_lower[-1]]))
        y_upper = int(1.1 * np.amax([train_upper[-1], test_upper[-1]]))
        axes.set_ylim((y_lower, y_upper))

    plt.show()
    plt.close()

def validation_curves(algorithm, formulas, data, values, evaluate, fold_count=10, repetitions=3, increment=1):
  indices = list(range(len(data)))
  results = data_collection()
  for _ in range(repetitions):
      random.shuffle(indices)
      folds = chunk(indices, fold_count)
      for fold in folds:
          test_data = data.iloc[ fold]
          train_indices = [idx for idx in indices if idx not in fold]
          train_data = data.iloc[train_indices]
          for i, p in enumerate(zip(formulas, values)):
              f, v = p
              # it's ok to resuse the folds for each v of values
              # we calculate the model
              result = algorithm(f, train_data, v)
              model = result["model"]
              # we calculate the results for the training data subset
              y, X = patsy.dmatrices(f, train_data, return_type="matrix")
              result = summarize(f, X, y, model)
              metric = evaluate(result)
              results["train"][i].append( metric)
              
              # we calculate the results for the test data.
              y, X = patsy.dmatrices(f, test_data, return_type="matrix")
              result = summarize(f, X, y, model)
              metric = evaluate(result)
              results["test"][i].append(metric)
  # process results
  # Rely on the CLT...
  statistics = {}
  for k, v in results["train"].items():
      statistics[ k] = (np.mean(v), np.std(v))
  results["train"] = statistics
  statistics = {}
  for k, v in results["test"].items():
      statistics[k] = (np.mean(v), np.std(v))
  results["test"] = statistics
  return results
  
def plot_validation_curves(results, metric, parameter, values, zoom=False):
    figure = plt.figure(figsize=(10,6))

    axes = figure.add_subplot(1, 1, 1)

    xs, train_lower, train_mean, train_upper = results_to_curves("train", results)
    _, test_lower, test_mean, test_upper = results_to_curves("test", results)

    axes.plot(values, train_mean, color="steelblue")
    axes.fill_between(values, train_upper, train_lower, color="steelblue", alpha=0.25, label="train")
    axes.plot(values, test_mean, color="firebrick")
    axes.fill_between(values, test_upper, test_lower, color="firebrick", alpha=0.25, label="test")
    axes.legend()
    axes.set_xlabel(parameter)
    axes.set_ylabel(metric)
    axes.set_title("Validation Curves")

    if zoom:
        y_lower = int(0.9 * np.amin([train_lower[-1], test_lower[-1]]))
        y_upper = int(1.1 * np.amax([train_upper[-1], test_upper[-1]]))
        axes.set_ylim((y_lower, y_upper))

    plt.show()
    plt.close()
	
def knn_cross_validation(formula, builder, data, fold_count=10, repetitions=3):
    indices = list(range(len(data)))
    metrics = {"train": [], "test": []}
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            # y, X for training data
            y, X = patsy.dmatrices(formula, train_data, return_type="matrix")
            model = builder.fit(X, y)
            y_hat = model.predict(X)
            training_r_squared = (stats.pearsonr(y, y_hat)[0][0])**2
            metrics["train"].append(training_r_squared)
            # y, X for training data
            y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
            y_hat = model.predict(X)
            test_r_squared = (stats.pearsonr(y, y_hat)[0][0])**2
            metrics["test"].append(test_r_squared)
    return metrics
	
def knn_learning_curves(builder, formula, data, fold_count=10, repetitions=3, increment=1):
    indices = list(range(len( data)))
    results = data_collection()
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[ fold]
            train_indices = [idx for idx in indices if idx not in fold]
            train_data = data.iloc[train_indices]
            for i in list(range(increment, 100, increment)) + [100]: # ensures 100% is always picked.
                # the indices are already shuffled so we only need to take ever increasing chunks
                train_chunk_size = int( np.ceil((i/100)*len( train_indices)))
                train_data_chunk = data.iloc[train_indices[0:train_chunk_size]]
                # we calculate the model
                y, X = patsy.dmatrices(formula, train_data_chunk, return_type="matrix")
                model = builder.fit(X, y)
                y_hat = model.predict(X)
                training_r_squared = (stats.pearsonr(y, y_hat)[0][0])**2
                results["train"][i].append(training_r_squared)
                
                # y, X for test data
                y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
                y_hat = model.predict(X)
                test_r_squared = (stats.pearsonr(y, y_hat)[0][0])**2
                results["test"][i].append(test_r_squared)
    # process results
    # Rely on the CLT...
    statistics = {}
    for k, v in results["train"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["train"] = statistics
    statistics = {}
    for k, v in results["test"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["test"] = statistics
    return results