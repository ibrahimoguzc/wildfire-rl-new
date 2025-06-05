# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/
"""Contains classes for defining SOSID simulation atmospheres."""

import math
from datetime import datetime, time, timedelta
from functools import lru_cache

from sosid.util.abc import abstractattribute


class BaseAtmosphere:
    """Define basic parameters needed to describe the Atmosphere."""

    def __init__(self, simulation):
        self.simulation = simulation

    @property
    def parameters(self):
        return self.simulation.parameters

    @property
    def mission_time(self) -> datetime:
        return self.simulation.timer.mission_time

    @abstractattribute
    def temperature(self) -> float:
        pass

    @abstractattribute
    def relative_humidity(self) -> float:
        pass

    @abstractattribute
    def wind_speed(self) -> float:
        pass

    @abstractattribute
    def wind_aspect(self) -> float:
        pass


class AtmosphereDayNight(BaseAtmosphere):
    """Add Day/Night cycle through sunset and sunrise times."""

    def __init__(self, simulation):
        super().__init__(simulation)

    @abstractattribute
    def next_sunset(self) -> datetime:
        """Return the next sunset time in datetime format."""

    @abstractattribute
    def next_sunrise(self) -> datetime:
        """Return the next sunrise time in datetime format."""


# TODO Implement the necessary logic to ensure that the model accounts
# for the fire map boundaries without leading segfaults.
class AtmosphereMathematical(AtmosphereDayNight):
    """Implements Mathematical Functions to model the Atmosphere."""

    def __init__(self, simulation):
        super().__init__(simulation)

    @property
    def temperature(self):
        """Returns the current temperature."""
        return self._temperature_from_func(self.cache_access)

    @property
    def relative_humidity(self):
        """Returns the current relative humidity."""
        return self._relative_humidity_from_func(self.cache_access)

    @property
    def wind_speed(self):
        """Returns the current wind speed."""
        return self._wind_speed_from_func(self.cache_access)

    @property
    def wind_aspect(self):
        """Returns the current wind aspect/direction."""
        return self._wind_aspect_from_func(self.cache_access)

    @property
    def cache_access(self):
        return int(
            self.simulation.timer.mission_runtime.total_seconds()
            / self.atmosphere_parameters.update_frequency
        )

    @property
    def time_in_hours(self):
        """Return current time in hours."""
        return self.mission_time.hour + self.mission_time.minute / 60

    @property
    def atmosphere_parameters(self):
        """Provide interface to input parameters."""
        return self.parameters.atmosphere_inputs

    @lru_cache(maxsize=1)
    def _temperature_from_func(self, cache_access):
        """Returns the current temperature value.

        Based on the inputs: min/max/min' temperature, the times these
        occur, sunrise/sunset time inputs fed by
        `atmosphere_parameters`.

        The scientific background can be found in the article "Modelling
        diurnal patterns of air temperature, radiation, windspeed and
        relative humidity by equations from daily characteristics",
        Ephrath et al. (1977)
        """
        sunrise, sunset = self.atmosphere_parameters.sun_times
        (
            min_temp_time,
            max_temp_time,
        ) = self.atmosphere_parameters.temperature_times
        (
            min_temp,
            max_temp,
            min_temp_next_day,
        ) = self.atmosphere_parameters.temperature_range
        t_k = 15  # a calibration parameter (°C) valid for several locs
        day_length = sunset - sunrise  # day length (sunset-sunrise)
        night_length = 24 - day_length  # night length
        time_max_solar = self.atmosphere_parameters.time_of_max_solar_height
        p = max_temp_time - time_max_solar
        tau = 4  # mean value from the viewed paper
        sinus = math.sin(
            math.pi
            * (
                (self.time_in_hours - time_max_solar + (day_length / 2))
                / (day_length + 2 * p)
            )
        )
        amp = (max_temp - min_temp) * (1 + ((max_temp - min_temp) / t_k))
        simple_sinus = math.sin(math.pi * (day_length / (day_length + 2 * p)))
        t_s = min_temp_next_day + (max_temp - min_temp_next_day) * simple_sinus
        if (
            self.time_in_hours >= min_temp_time
            and self.time_in_hours <= sunset
        ):
            curr_temp = (
                min_temp
                - (t_k / 2)
                + (1 / 2) * math.sqrt((t_k**2) + 4 * amp * t_k * sinus)
            )
        elif self.time_in_hours < min_temp_time or self.time_in_hours > sunset:
            if self.time_in_hours < min_temp_time:
                time_in_hours = self.time_in_hours + 24
            else:
                time_in_hours = self.time_in_hours
            curr_temp = (
                min_temp_next_day
                - t_s * math.e ** (-(night_length / tau))
                + (t_s - min_temp_next_day)
                * math.e ** (-((time_in_hours - sunset) / tau))
            ) / (1 - math.e ** (-(night_length / tau)))
        temperature = round(curr_temp, 2)
        return temperature

    @lru_cache(maxsize=1)
    def _relative_humidity_from_func(self, cache_access):
        """Returns the current relative humidity value.

        Based on: min/max, rel. humidity,  min/max air temperature, and
        current temperature inputs fed by `atmosphere_parameters`.

        The scientific background can be found in the article "An
        integrated evaluation of thirteen modelling solutions
        for the generation of hourly values of air relative humidity",
        Bregaglio et al. (2010). The model nr. 7 is utilized here,
        because it requires the least inputs without further assumptions
        to be made.
        """
        curr_temp = self.temperature
        min_humidity, max_humidity = self.atmosphere_parameters.humidity_range
        min_temp, max_temp, _ = self.atmosphere_parameters.temperature_range
        rh_flt = max_humidity + (
            (curr_temp - min_temp)
            / (max_temp - min_temp)
            * (min_humidity - max_humidity)
        )
        relative_humidity = int(rh_flt)
        return relative_humidity

    @lru_cache(maxsize=1)
    def _wind_speed_from_func(self, cache_access):
        """Returns a current windspeed value.

        Based on: total daily windrun (km/d), times of sunrise/sunset
        inputs fed by `atmosphere_parameters`.

        The scientific background can be found in the article "Modelling
        diurnal patterns of air temperature, radiation, windspeed and
        relative humidity by equations from daily characteristics",
        Ephrath et al. (1977). However, the explained function could not
        be brought to work. An approximation is used but it is not
        scientifically validated. Soon to be revisited.
        """
        sun_rise, sunset = self.atmosphere_parameters.sun_times
        wind_run = self.atmosphere_parameters.wind_run
        t1 = sun_rise + 1  # time constant nr 1 (right after sunrise)
        t2 = 12 + 3  # time constant nr 2 (2-4 hrs after noon)
        t3 = sunset + 2  # time constant nr 2 (0-2 hrs after sunset)
        tw1 = t1  # time interval factor nr 1
        sf1 = 4 * (t2 - t1)  # time interval factor nr 3
        sf2 = 4 * (t3 - t2)  # time interval factor nr 4
        ratio = 0.0080
        wind_min = wind_run * ratio
        wind_max = ((wind_run - wind_min * 24 * 3.6) * 2 * math.pi * 1000) / (
            3600 * (sf1 + sf2)
        )
        if self.time_in_hours >= t1 and self.time_in_hours <= t3:
            curr_windspeed = wind_min + wind_max * math.sin(
                2 * math.pi * ((self.time_in_hours - tw1) / sf1)
            )
        else:
            curr_windspeed = wind_min
        wind_speed = round(curr_windspeed, 2)
        return wind_speed

    @lru_cache(maxsize=1)
    def _wind_aspect_from_func(self, cache_access):
        """Returns a random winddirection.

        Based on: General wind direction tendency and a varying range
        defined by inputs fed by `atmosphere_parameters`.

        This model has no scientific background since winddirection is
        highly dependent on the actual weather situation of the
        surrounding area of a much greater scale than the investigated
        area.
        """
        wind_dir_top = (
            self.atmosphere_parameters.general_winddirection
            + self.atmosphere_parameters.range_winddirection
        )
        wind_dir_bottom = (
            self.atmosphere_parameters.general_winddirection
            - self.atmosphere_parameters.range_winddirection
        )
        curr_wind_dir = self.simulation.random.uniform(
            wind_dir_bottom, wind_dir_top
        )
        wind_aspect = round(curr_wind_dir, 2)
        return wind_aspect

    @property
    def next_sunset(self) -> datetime:
        """Return the next sunset time in datetime format."""
        sunset_hour = self.atmosphere_parameters.sun_times[1]
        date = self.mission_time.date()
        sunset_time = datetime.combine(date, time(hour=int(sunset_hour)))
        # Check whether sun has set for current day
        if self.mission_time > sunset_time:
            # Return sun set time of next day
            date = (self.mission_time + timedelta(days=1)).date()
        return datetime.combine(date, time(hour=int(sunset_hour)))

    @property
    def next_sunrise(self) -> datetime:
        """Return the next sunrise time in datetime format."""
        sunrise_hour = self.atmosphere_parameters.sun_times[0]
        date = self.mission_time.date()
        sunrise_time = datetime.combine(date, time(hour=int(sunrise_hour)))
        # Check whether sun has risen for current day
        if self.mission_time > sunrise_time:
            # Return sun rise time of next day
            date = (self.mission_time + timedelta(days=1)).date()
        return datetime.combine(date, time(hour=int(sunrise_hour)))
