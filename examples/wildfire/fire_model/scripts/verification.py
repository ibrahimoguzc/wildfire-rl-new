# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""This file is used to verify the behavior of the fire models."""

import os

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from examples.wildfire.fire_model.jit_funcs.cpu import step as step_cpu
from examples.wildfire.fire_model.preallocate import preallocate_cpu
from examples.wildfire.fire_model.states import BURNT, CMAP, FULL_BURNING
from examples.wildfire.paths import FIGURE_DIR


class PropagationVerification:
    STEP_SIZE_FACTORS = [0.01, 0.03, 0.1, 0.125, 0.3, 0.5, 1.0, 2.0]
    N_ITERS = [1600, 600, 230, 200, 130, 90, 47, 45]
    CELL_SIZE = 30
    CORRECTION_COEFFICIENT = 1
    INPUT_DATA = {
        "shape": (128, 128),
        "ambient_temperature": 15,
        "wind_speed": 0,
        "wind_aspect": 0,
        "relative_humidity": 35,
        "terrain_slope": 0,
        "terrain_aspect": 0,
        "avg_combustibility": 1,
        "stochastic": False,
        # float32 is faster than float64
        # (Anaconda introduction to GPU programming)
        "dtype": np.float64,
    }

    @property
    def fire_origin(self):
        """Computes the origin of the fire based on the grid size."""
        return tuple(idx // 2 for idx in self.INPUT_DATA["shape"])

    def run_cpu(self):
        """Runs the CPU fire-model for each step size factor."""
        results, input_data = {}, self.INPUT_DATA

        for factor, n_iters in zip(self.STEP_SIZE_FACTORS, self.N_ITERS):
            arrays = preallocate_cpu(**input_data)
            arrays["fire_states"][self.fire_origin] = FULL_BURNING
            arrays["fire_indices"][0] = self.fire_origin
            n_burning = 1
            for _ in range(n_iters):
                n_burning = step_cpu(
                    0,
                    **arrays,
                    cell_size=self.CELL_SIZE,
                    correction_coefficient=self.CORRECTION_COEFFICIENT,
                    step_size_factor=factor,
                    n_burning=n_burning,
                )

            results[factor] = arrays["fire_states"]
        return results

    def plot_cpu(self):
        """Plots the results of the CPU fire model."""
        fig = plt.figure("FireModelVerification")
        fig, axes = plt.subplots(
            nrows=2,
            ncols=4,
            num="FireModelVerification",
            gridspec_kw={
                "top": 0.9,
                "wspace": 0.1,
                "hspace": -0.4,
                "bottom": 0.1,
            },
        )
        axes = axes.flatten()
        results = self.run_cpu()
        subfig_label = "abcdefgh"
        for i, (factor, fire_states) in enumerate(results.items()):
            ax = axes[i]
            ax.imshow(fire_states, vmin=0, vmax=BURNT, cmap=CMAP)
            circle = patches.Circle(
                self.fire_origin,
                radius=min(self.INPUT_DATA["shape"]) * 0.35,
                edgecolor="w",
                fill=False,
                linestyle="--",
            )
            ax.add_patch(circle)
            ax.text(1, 17.5, f"({subfig_label[i]})", color="w")
            ax.text(
                self.INPUT_DATA["shape"][1] * 0.25,
                self.INPUT_DATA["shape"][0] * 0.975,
                f"$m={factor}$",
                color="w",
            )
            ax.axis("off")
        plt.show()
        fig.savefig(
            fname=os.path.join(FIGURE_DIR, f"{fig.get_label()}.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        return "Figure Plotted and Saved"


if __name__ == "__main__":
    obj = PropagationVerification()
    obj.plot_cpu()
