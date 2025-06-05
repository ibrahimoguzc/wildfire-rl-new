# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

# Plot all weather params with time
# For API and Mathematical

from random import Random

import matplotlib.pyplot as plt

from examples.wildfire.paths import FIGURE_DIR
from examples.wildfire.simulation import AtmosphereParametersMathematical
from src.sosid.environment.atmosphere import (
    AtmosphereMathematical,
    BaseAtmosphere,
)

ATMOSPHERE_MATHEMATICAL_INPUTS = {
    "temperature_range": (10.0, 25.0, 10.0),
    # times when temperature is lowest/highest
    "temperature_times": (5.0, 15.0),
    # sun[0] = sunrise // sun[1] = sunset in hrs
    "sun_times": (6.0, 20.0),
    # Time sun is at the highest peak in hrs
    "time_of_max_solar_height": 12,  # Location specific
    # minimum and maximum relative humidity in %
    "humidity_range": (20.0, 70.0),
    # total windrun of the day in km/d
    "wind_run": 500.0,
    # overall tendency of winddirection
    "general_winddirection": 300,
    # range in which the wind direction varies randomly
    "range_winddirection": 30,
    # Freuqncy with which to update the atmospheric parameteres
    "update_frequency": 600,  # SI seconds
}


class SimPlaceholder:
    def __init__(self, random):
        self.random = random


class BasePlotClass(BaseAtmosphere):
    def plot(self, title, filename):
        hours = []
        temperatures = []
        humidities = []
        wind_speeds = []
        wind_aspects = []
        for hour in range(24):
            self._time_in_hours = hour
            hours.append(hour)
            temperatures.append(self.temperature)
            humidities.append(self.relative_humidity)
            wind_speeds.append(self.wind_speed)
            wind_aspects.append(self.wind_aspect)
        plt.style.use("seaborn-v0_8")
        fig = plt.figure()
        fig.suptitle(title, fontsize=14)
        fig.set_figheight(4)
        fig.set_figwidth(8)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(hours, temperatures)
        ax1.set_ylabel("Temperature, °C")
        ax1.set_xlabel("Time, hr")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(hours, humidities)
        ax2.set_ylabel("Relative Humidity, %")
        ax2.set_xlabel("Time, hr")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(hours, wind_speeds)
        ax3.set_ylabel("Wnd Speeds, m/s")
        ax3.set_xlabel("Time, hr")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(hours, wind_aspects)
        ax4.set_ylabel("Wind Aspect, °")
        ax4.set_xlabel("Time, hr")

        plt.show()
        fig.savefig(
            fname=str(FIGURE_DIR / filename),
            format="pdf",
            bbox_inches="tight",
        )
        return "Figure Plotted and Saved"


class PlotAtmosphereMathematical(AtmosphereMathematical, BasePlotClass):
    def __init__(self):
        self._time_in_hours = 0
        self._atmosphere_parameters = AtmosphereParametersMathematical(
            **ATMOSPHERE_MATHEMATICAL_INPUTS
        )
        random = Random(1)
        self.simulation = SimPlaceholder(random)

    @property
    def time_in_hours(self):
        return self._time_in_hours

    @property
    def atmosphere_parameters(self):
        return self._atmosphere_parameters

    @property
    def cache_access(self):
        return self._time_in_hours


if __name__ == "__main__":
    filename = "AtmosphereMathematicalVerification.pdf"
    title = "Modelling Dynamic Atmosphere through Mathematical Functions"
    PlotAtmosphereMathematical().plot(title, filename)
