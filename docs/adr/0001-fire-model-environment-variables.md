# Fire Model Environment Conditions

* Status: Proposed
* Deciders: San Kilkis
* Date: 2019-09-05

## Context and Problem Statement

The fire-model of Rui, 2018 was selected due to its ability to incorporate
physical quantities such as combustibles, wind, temperature, humidity, and
slope. However, the question remains of whether to implement these as a
constant value globally across the entire cellular space, or to have different
values in each cell.

## Decision Drivers

* Ability to later implement actual gridded-data of recorded wildfires
* Performance, the update of environment conditions shouldn't be slower than
  the run-time required to compute propagation directions

## Considered Options

* Constant value used in propagation calculations
* An array of constant values (duplicated)
* Creating an array w/ noise applied to all cells

## Decision Outcome

Chosen option: "An array of constant values (duplicated)", because it
accomplishes the goal of being able to later implement actual gridded data
while maintaining an acceptable performance. Although, memory use will
increase, the size of the arrays at around (1000x1000) cells results in memory
consumption in the MB range which isn't too bad. Finally, the option of
creating an array with noise applied to all cells, was not chosen as
arbitrarily adding noise to the data-set without physical basis does not
increase the fidelity of the model.

### Positive Consequences

* Ability to later implement actual gridded data

### Negative Consequences

* Increased memory consumption from having to store extra arrays with
  currently no added information, as well as a performance hit from
  needing to re-compute the initial forest-fire spread, R0, at each cell.

## Pros and Cons of the Options

### Constant value used in propagation calculations

A single integer or float value is used to change the environmental
conditions of the wildfire. Therefore, at each discrete time-step the
initial forest fire spread, R0, needs to be computed once.

* Good, because it is very simple and requires little memory to implement
* Performance is great since only an int or float needs to be overwritten
* Bad, because future verification attempts with actual data sets will
  require modification to the code

### An array of constant values (duplicated)

A single integer or float value is mapped to an array of the same size as
the cellular space. At each discrete time-step this requires the initial
forest fire spread, R0, to be re-computed for each cell (unless caching
is used).

* Good, because it allows for the usage of actual data-sets later on
* Bad, because it requires additional overhead to re-compute R0 for each
  cell

### Creating an array w/ noise applied to all cells

A single integer or float value is first mapped to an array of the same
size as the cellular space and then a noise array is constructed utilizing
the numpy.random.normal function. The noise array is then added to the
value array in-place.

* Bad, because it requires ~100 ms of runtime for a 1000x1000 array which
  means the propagation direction calculation is faster than the numpy
  np.random.normal function
* Bad, because without a physical basis for applying variance to the mean
  environmental quantities, the fidelity of the model is not imporved.

## Links

<!-- TODO add link to Fire model of wang -->
* [Link type] [Link to ADR] <!-- example: Refined by [ADR-0005](0005-example.md) -->
* … <!-- numbers of links can vary -->
