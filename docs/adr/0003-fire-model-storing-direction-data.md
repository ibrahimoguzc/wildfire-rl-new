# Fire Model Storing Direction Data

* Status: Proposed
* Deciders: San Kilkis
* Date: 2019-09-05

## Context and Problem Statement

The fire-model of Rui, 2018 previously selected by [ADR-0001] requires the
use of local direction quantities to determine the local rate of fire-spread.
Therefore, for each cell of the CA fire-model a propagation direction, wind
direction and slope direction is required. The problem is to find a way to
represent these quantities while minimizing memory usage, maximizing
readability, and making sure that the chosen format is compatible with tools
like WindNinja and terrain slope output from tools that use DEM.

## Decision Drivers

* Memory Usage
* Readability
* Compatibility

## Considered Options

* Use 3D array where the j-th has length 2 corresponding to a 2D vector
* Use 2D array with direction provided in degrees 0 to 360 (aspect)

## Decision Outcome

Chosen option: Use 2D array with direction provided in degrees 0 to 360
(aspect)

This decision results in a minimization of memory usage for each direction
array (wind, slope, propagation direction) as well as makes the values more
human redable. The downside is that `math.nan` values are required to express
the 0 vector, and one must be very careful with usage of sines and cosines with
respect to the current quadrant. Therfore, using a 2D array with direction
provided by degrees will require detailed testing. However, most importantly,
using a direction specified by an aspect between 0-360 will adhere to both
meteorological convention as adopted by [WindNinja] and convention used for
expressing terrain slope direction as adopted by [ArcGIS].

### Positive Consequences

* Memory usage for storing direction information is decreased
* Understanding the direciton data is easier
* Direction data is more compatible with wind and terrain tools

### Negative Consequences

* Requires additional testing to ensure calculations work in all quadrants
* Requires use of `math.nan` values to express the zero vector

## Pros and Cons of the Options

### Use 3D array where the j-th has length 2 corresponding to a 2D vector

* Good, because direction vectors can be directly used without the need
  of transcendental functions such as `math.sin` and `math.cos`.
* Good, because it is more generalizable and easier to test vectorial math
  works irrespective of the quadrant within a unit-circle.
* Bad, because memory usage is increased due to having to store 2 `float`
  values for each direction variable.

### Use 2D array with direction provided in degrees 0 to 360 (aspect)

* Good, because it minimizes memory usage by only requiring 1 `float`
* Good, because it is easier to read
* Good, because there are less variables to unpack
* Bad, because it cannot represent the zero vector without using `math.nan`
* Bad, because it requires additional testing to ensure correct output in all
  4 quadrants of the unit-circle.

<!-- Un-wrapped Text Below for References, Links, Images, etc. -->
[ADR-0001]: 0001-fire-model-environment-variables.md
[WindNinja]: http://firelab.github.io/windninja/pdf/WindNinja_tutorial3.pdf
[ArcGIS]: http://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-aspect-works.htm
