import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.integrate import odeint
import lmfit
from lmfit.model import save_modelresult, load_modelresult

import warnings
import yaml

warnings.filterwarnings("ignore")

# read the configuration
with open(r"config.yml") as conf:
    config = yaml.load(conf, Loader=yaml.FullLoader)
    conf.close()

disease = config["train"]["disease_name"]

with open(r"models/" + config["model_file"]) as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    file.close()

store_file = data["model_constants"]["lmfit_model"]

R_0 = data["train"][disease]["fixed_parameters"]["R_0"]
Infection_period = data["train"][disease]["fixed_parameters"]["infection_period"]
Incubation_period = data["train"][disease]["fixed_parameters"]["incubation_period"]
alpha = data["train"][disease]["fixed_parameters"]["death_rate"]
days_till_death = data["train"][disease]["fixed_parameters"]["days_until_death"]

# read the data
agegroups = pd.read_csv("data/agegroups.csv")
probabilities = pd.read_csv("data/probabilities.csv")
covid_data = pd.read_csv(
    "data/time_series_covid19_deaths_global_narrow.csv",
    parse_dates=["Date"],
    skiprows=[1],
)
covid_data["Location"] = covid_data["Country/Region"]

# create some dicts for fast lookup
# 1. agegroups
agegroup_lookup = dict(
    zip(
        agegroups["Location"],
        agegroups[
            [
                "0_9",
                "10_19",
                "20_29",
                "30_39",
                "40_49",
                "50_59",
                "60_69",
                "70_79",
                "80_89",
                "90_100",
            ]
        ].values,
    )
)

starting_infected = 1  # how many people start infected
starting_dead = 0  # how many people start dead
how_long = 2000  # how many days for simulation to run

# these variables are the ones that we put into the models
gamma = 1.0 / Infection_period  # infection period
delta = 1.0 / Incubation_period  # incubation period
rho = 1 / days_till_death  # days from infection until death

plt.gcf().subplots_adjust(bottom=0.15)


def plotter(t, S, E, I, R, D, R_0, B, S_1=None, S_2=None, x_ticks=None):

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

    plt.show()

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
    newDs = [0] + [D[i] - D[i - 1] for i in range(1, len(t))]
    if x_ticks is None:
        ax2.plot(t, newDs, "r--", alpha=0.7, linewidth=2, label="total")

    else:
        ax2.plot(x_ticks, newDs, "r--", alpha=0.7, linewidth=2, label="total")
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_minor_locator(mdates.MonthLocator())
        f.autofmt_xdate()

    ax2.title.set_text("Deaths per day")
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
    N,
    mask_effectiveness_in,
    mask_effectiveness_out,
    compliance,
):
    S, E, I, R, D = y

    epsilon = (1 - mask_effectiveness_in * compliance) * (
        1 - mask_effectiveness_out * compliance
    )

    dSdt = -beta(t) * S * I * epsilon / N
    dEdt = beta(t) * S * I * epsilon / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I

    return dSdt, dEdt, dIdt, dRdt, dDdt


# gamma = 1.0 / 9.0
# sigma = 1.0 / 3.0

"""
# DEPRECIATED: not used
def logistic_R_0(t, R_0_start_mult, k, x0, R_0_end_mult):
    return (R_0_start_mult - R_0_end_mult) / (1 + np.exp(-k * (-t + x0))) + R_0_end_mult
"""


def double_logistic_R_0(
    t, R_0, R_0_start_mult, k1, x01, R_0_middle_mult, k2, x02, R_0_end_mult
):
    return (
        (R_0_start_mult * R_0 - R_0_middle_mult * R_0) / (1 + np.exp(-k1 * (-t + x01)))
        + R_0_middle_mult * R_0
    ) * (
        (R_0_middle_mult * R_0 - R_0_end_mult * R_0) / (1 + np.exp(-k2 * (-t + x02)))
        + R_0_end_mult * R_0
    )


def Model(
    days,
    population_size,
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
):
    def beta(t):
        return (
            double_logistic_R_0(
                t, R_0, R_0_start_mult, k1, x01, R_0_middle_mult, k2, x02, R_0_end_mult
            )
            * gamma
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
            N,
            mask_effectiveness_in,
            mask_effectiveness_out,
            compliance,
        ),
    )
    S, E, I, R, D = ret.T
    R_0_over_time = [beta(i) / gamma for i in range(len(t))]

    return (
        t,
        S,
        E,
        I,
        R,
        D,
        R_0_over_time,
        compliance,
        mask_effectiveness_out,
        mask_effectiveness_in,
    )


# parameters
data = covid_data[covid_data["Location"] == "Italy"]["Value"].values[::-1]
population_size = sum(agegroup_lookup["Italy"])
# beds_per_100k = beds_lookup["Italy"]
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
):
    ret = Model(
        days,
        population_size,
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
    )
    return ret[5][x]


def fit(fitter, days, data, population_size, params_init_min_max, store_file):

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
    first_date = np.datetime64(covid_data.Date.min()) - np.timedelta64(
        outbreak_shift, "D"
    )
    x_ticks = pd.date_range(start=first_date, periods=full_days, freq="D")

    plotter(
        *Model(
            full_days,
            population_size,
            **result.best_values,
        ),
        x_ticks=x_ticks,
    )

    save_modelresult(result, store_file)
    return result.best_values


"""
def predict(days, data, population_size, params_init_min_max, store_file):
    plotter(
        *Model(
            full_days,
            population_size,
            **result.best_values,
        ),
        x_ticks=x_ticks,
    )
"""

x = fit(fitter, days, data, population_size, params_init_min_max, store_file)
print(x)