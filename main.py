import numpy as np
import pandas as pd

# from operator import itemgetter

pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.integrate import odeint
import lmfit

# from lmfit.model import save_modelresult, load_modelresult

import warnings
import yaml
from pathlib import Path
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")

config_path = Path.cwd() / "config.yml"
# read the configuration
with open(config_path, "r") as conf:
    config = yaml.load(conf, Loader=yaml.FullLoader)
    conf.close()

disease = config["train"]["disease_name"]
test_disease = config["test"]["disease_name"]

model = Path.cwd() / "models" / config["model_file"]

with open(model, "r") as file:
    mod_data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
# store_file = mod_data["model_constants"]["lmfit_model"]

R_0 = mod_data["train"][disease]["fixed_parameters"]["R_0"]
Infection_period = mod_data["train"][disease]["fixed_parameters"]["infection_period"]
Incubation_period = mod_data["train"][disease]["fixed_parameters"]["incubation_period"]
alpha = mod_data["train"][disease]["fixed_parameters"]["death_rate"]
days_till_death = mod_data["train"][disease]["fixed_parameters"]["days_until_death"]
how_long = mod_data["train"]

country_name = mod_data["model_constants"]["name"]

covid_data = pd.read_csv(
    Path("data/owid-covid-data.csv"),
)

pop_size_data = pd.read_csv(Path("data/projected-population-by-country.csv"))


starting_infected = 1  # how many people start infected
starting_dead = 0  # how many people start dead

# these variables are the ones that we put into the models
gamma = 1.0 / Infection_period  # infection period
delta = 1.0 / Incubation_period  # incubation period
rho = 1 / days_till_death  # days from infection until death


def plotter(t, S, E, I, R, D, R_0, compliance, S_1=None, S_2=None, x_ticks=None):

    f, ax = plt.subplots(1, 1, figsize=(20, 4))
    if x_ticks is None:
        # ax.plot(t, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
        ax.plot(t, E, "y", alpha=0.7, linewidth=2, label="Exposed")
        ax.plot(t, I, "r", alpha=0.7, linewidth=2, label="Infected")
        ax.plot(t, R, "g", alpha=0.7, linewidth=2, label="Recovered")
        ax.plot(t, D, "k", alpha=0.7, linewidth=2, label="Dead")
    else:
        # ax.plot(x_ticks, S, "b", alpha=0.7, linewidth=2, label="Susceptible")
        ax.plot(x_ticks, E, "y", alpha=0.7, linewidth=2, label="Exposed")
        ax.plot(x_ticks, I, "r", alpha=0.7, linewidth=2, label="Infected")
        ax.plot(x_ticks, R, "g", alpha=0.7, linewidth=2, label="Recovered")
        ax.plot(x_ticks, D, "k", alpha=0.7, linewidth=2, label="Dead")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        f.autofmt_xdate()

    ax.title.set_text("SEIR-Model with double logistic curves")

    ax.grid(b=True, which="major", c="w", lw=2, ls="-")
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    f = plt.figure(figsize=(20, 4))

    # sp1
    ax1 = f.add_subplot(131)
    if x_ticks is None:
        ax1.plot(t, R_0, "b--", alpha=0.7, linewidth=2, label="R_0")
    else:
        ax1.plot(x_ticks, R_0, "b--", alpha=0.7, linewidth=2, label="R_0")
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        f.autofmt_xdate()

    ax1.title.set_text("R_0 over time")
    ax1.grid(b=True, which="major", c="w", lw=2, ls="-")
    legend = ax1.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    # sp2
    ax2 = f.add_subplot(132)
    if x_ticks is None:
        ax2.plot(t, compliance, "r--", alpha=0.7, linewidth=2, label="total")

    else:
        ax2.plot(x_ticks, compliance, "r--", alpha=0.7, linewidth=2, label="total")
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_minor_locator(mdates.MonthLocator())
        f.autofmt_xdate()

    ax2.title.set_text("Compliance over time")
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_tick_params(length=0)
    ax2.grid(b=True, which="major", c="w", lw=2, ls="-")
    legend = ax2.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    # sp3
    ax3 = f.add_subplot(133)

    if x_ticks is None:
        ax3.plot(t, D, "b--", alpha=0.7, linewidth=2, label="Deaths")
    else:
        ax3.plot(x_ticks, D, "b--", alpha=0.7, linewidth=2, label="Deaths")
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax3.xaxis.set_minor_locator(mdates.MonthLocator())
        f.autofmt_xdate()

    ax3.title.set_text("Total deaths over time")
    ax3.grid(b=True, which="major", c="w", lw=2, ls="-")
    legend = ax1.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    plt.show()


def deriv(
    y,
    t,
    beta,
    gamma,
    delta,
    rho,
    alpha,
    epsilon,
    N,
):
    S, E, I, R, D = y

    dSdt = -beta(t) * S * I * epsilon(t) / N
    dEdt = beta(t) * S * I * epsilon(t) / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I

    return dSdt, dEdt, dIdt, dRdt, dDdt


def logistic(t, l, l_start_mult, k, x0, l_end_mult):
    return (l_start_mult * l - l_end_mult * l) / (
        1 + np.exp(-k * (-t + x0))
    ) + l_end_mult * l


def double_logistic(t, l, l_start_mult, k1, x01, l_middle_mult, k2, x02, l_end_mult):
    return (
        (l_start_mult * l - l_middle_mult * l) / (1 + np.exp(-k1 * (-t + x01)))
        + l_middle_mult * l
    ) * (
        (l_middle_mult * l - l_end_mult * l) / (1 + np.exp(-k2 * (-t + x02)))
        + l_end_mult * l
    )


def Model(
    days,
    population_size,
    gamma,
    delta,
    rho,
    alpha,
    R_0,
    R_0_start_mult,
    k1,
    x01,
    R_0_middle_mult,
    k2,
    x02,
    R_0_end_mult,
    mask_effectiveness_in,
    mask_effectiveness_out,
    compliance,
    compliance_start_mult,
    compliance_k,
    compliance_x0,
    compliance_end_mult,
):
    def beta(t):
        return (
            double_logistic(
                t, R_0, R_0_start_mult, k1, x01, R_0_middle_mult, k2, x02, R_0_end_mult
            )
            * gamma
        )

    def compliance_t(t):
        return logistic(
            t,
            compliance,
            compliance_start_mult,
            compliance_k,
            compliance_x0,
            compliance_end_mult,
        )

    def epsilon(t):

        return (1 - mask_effectiveness_in * compliance_t(t)) * (
            1 - mask_effectiveness_out * compliance_t(t)
        )

    N = population_size

    y0 = N - 1.0, 1.0, 0.0, 0.0, 0.0  # , 0.0
    t = np.linspace(0, days - 1, days)
    ret = odeint(
        deriv,
        y0,
        t,
        args=(
            beta,
            gamma,
            delta,
            rho,
            alpha,
            epsilon,
            N,
        ),
    )
    S, E, I, R, D = ret.T
    R_0_over_time = [beta(i) / gamma for i in range(len(t))]
    compliance_over_time = [compliance_t(i) for i in range(len(t))]

    return (t, S, E, I, R, D, R_0_over_time, compliance_over_time)


# parameters
data = covid_data[covid_data["location"] == country_name]["total_deaths"]
population_size = int(
    pop_size_data[pop_size_data["Entity"] == country_name][
        pop_size_data["Year"] == 2020
    ][
        "Population by country and region, historic and projections (Gapminder, HYDE & UN)"
    ]
)

outbreak_shift = 30
params_init_min_max = {
    "R_0_start_mult": (1.0, 0.3, 5.0),
    "k1": (2.5, 0.01, 5.0),
    "x01": (90, 0, 220),
    "R_0_middle_mult": (1.0, 0.3, 5.0),
    "k2": (2.5, 0.01, 5.0),
    "x02": (200, 0, 360),
    "R_0_end_mult": (1.0, 0.3, 5.0),
    "mask_effectiveness_in": (0.5, 0.0, 1.0),
    "mask_effectiveness_out": (0.5, 0.0, 1.0),
    "compliance": (0.6, 0.0, 1.0),
    "compliance_start_mult": (1.0, 0.0, 1.0),
    "compliance_k": (1.0, 0.001, 10.0),
    "compliance_x0": (90, 0, 300),
    "compliance_end_mult": (1.0, 0.0, 1.0),
}  # form: {parameter: (initial guess, minimum value, max value)}

days = outbreak_shift + len(data)


def fitter(
    x,
    R_0_start_mult,
    k1,
    x01,
    R_0_middle_mult,
    k2,
    x02,
    R_0_end_mult,
    mask_effectiveness_in,
    mask_effectiveness_out,
    compliance,
    compliance_start_mult,
    compliance_k,
    compliance_x0,
    compliance_end_mult,
):
    ret = Model(
        days,
        population_size,
        gamma,
        delta,
        rho,
        alpha,
        R_0,
        R_0_start_mult,
        k1,
        x01,
        R_0_middle_mult,
        k2,
        x02,
        R_0_end_mult,
        mask_effectiveness_in,
        mask_effectiveness_out,
        compliance,
        compliance_start_mult,
        compliance_k,
        compliance_x0,
        compliance_end_mult,
    )
    return ret[5][x]


def fit(
    fitter,
    days,
    data,
    population_size,
    params_init_min_max,
    gamma,
    delta,
    rho,
    alpha,
    R_0,
    model_file,
):

    if outbreak_shift >= 0:
        y_data = np.concatenate((np.zeros(outbreak_shift), data))
    else:
        y_data = y_data[-outbreak_shift:]

    x_data = np.linspace(
        0, days - 1, days, dtype=int
    )  # x_data is just [0, 1, ..., max_days] array

    mod = lmfit.Model(fitter)

    for kwarg, (init, mini, maxi) in params_init_min_max.items():
        mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

    params = mod.make_params()
    fit_method = "least_squares"

    y_data = np.nan_to_num(y_data)
    x_data = np.nan_to_num(x_data)

    result = mod.fit(y_data, params, method=fit_method, x=x_data)
    print(result.fit_report())
    result.plot_fit(datafmt="-")

    full_days = 500
    first_date = np.datetime64(covid_data.date.min())  # - np.timedelta64(
    #    outbreak_shift, "D"
    # )
    x_ticks = pd.date_range(start=first_date, periods=full_days, freq="D")

    modPredictScore = Model(
        len(y_data),
        population_size,
        gamma,
        delta,
        rho,
        alpha,
        R_0,
        **result.best_values,
    )

    pred_y_data = modPredictScore[5]

    r2 = float(r2_score(y_data, pred_y_data))

    print("The accuracy (R2 metric) for this model is: ", r2)

    plotter(
        *Model(
            full_days,
            population_size,
            gamma,
            delta,
            rho,
            alpha,
            R_0,
            **result.best_values,
        ),
        x_ticks=x_ticks,
    )

    with open(model_file, "w") as file:
        for param in result.best_values:
            mod_data["flexible_params"][param] = float(result.best_values[param])
        mod_data["accuracy"] = r2
        yaml.dump(mod_data, file)
    # return result.best_values


def predict(model_file):
    with open(model_file, "r") as f:
        mod = yaml.load(f, Loader=yaml.FullLoader)
    flex_params = {
        "R_0_start_mult": 0,
        "k1": 0,
        "x01": 0,
        "R_0_middle_mult": 0,
        "k2": 0,
        "x02": 0,
        "R_0_end_mult": 0,
        "mask_effectiveness_in": 0,
        "mask_effectiveness_out": 0,
        "compliance": 0,
        "compliance_start_mult": 0,
        "compliance_k": 0,
        "compliance_x0": 0,
        "compliance_end_mult": 0,
    }

    days = mod["model_constants"]["how_many_days"]

    disease_params = mod["test"][test_disease]
    for param in mod["flexible_params"]:
        flex_params[param] = mod["flexible_params"][param]

    R_0 = disease_params["R_0"]
    Infection_period = disease_params["infection_period"]
    Incubation_period = disease_params["incubation_period"]
    alpha = disease_params["death_rate"]
    days_till_death = disease_params["days_until_death"]

    # population_size = population_size

    # these variables are the ones that we put into the models
    gamma = 1.0 / Infection_period  # infection period
    delta = 1.0 / Incubation_period  # incubation period
    rho = 1 / days_till_death  # days from infection until death

    # print(flex_params)
    # print(disease_params)

    modelData = Model(
        days,
        population_size,
        gamma,
        delta,
        rho,
        alpha,
        R_0,
        **flex_params,
    )

    dfData = pd.DataFrame(
        modelData[1:6],
        index=["Susceptible", "Exposed", "Infected", "Recovered", "Dead"],
    )
    fileName = mod["model_constants"]["name"] + "_" + test_disease + ".csv"
    if config["save_data"] == True:
        dfData.to_csv(fileName, sep=",", encoding="utf-8")

    plotter(
        *modelData,
        x_ticks=None,
    )


def askFitOrPredict():
    inp = input(
        "Please type fit (f) or predict (p) depending on what you want to do. \n\
The config.yml file has an incorrect input for the option fit_or_predict \
and therefore requrires user input.\n[fit(f)/predict(p)]:"
    )
    if inp == "fit" or inp == "f":
        return "fit"
    elif inp == "predict" or inp == "p":
        return "predict"
    elif inp != "fit" or inp != "f" or inp != "predict" or inp != "p":
        print("That input was not recognised. Please try again.\n")
        askFitOrPredict()


def fitOrPredict(uinput):
    if uinput == "fit":
        fit(
            fitter,
            days,
            data,
            population_size,
            params_init_min_max,
            gamma,
            delta,
            rho,
            alpha,
            R_0,
            model,
        )
    elif uinput == "predict":
        predict(model)
    elif uinput != "fit" and uinput != "predict":
        fitOrPredict(askFitOrPredict())


fitOrPredict(config["fit_or_predict"])
