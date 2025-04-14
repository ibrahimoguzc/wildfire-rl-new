# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]. Currently, the project does not have a public API
signified by major version 0. Therefore, the API is subject to frequent
changes and backwards compatibility is not guaranteed.

## [0.18.0] - 2020-06-12

### Added

- Simulation time as well as pause and resume buttons to the `SoSID Viewer`
  (Closes #25) **PR #30 @MZupa**
- Agent fire-front suppression communication that allows `SuppressionUAV`
  agents to immediately reselect a firefront when the previous firefront has
  been extinguished **PR #48 @MZupa**
- Mission completion check in the `CPUFireModel` **PR #49 @MZupa**
- Mission report that prints in the console at the end of a `Simulation`
  **PR #49 @MZupa**
- `Simulation.stop` method to terminate a simulation through use of the
  `is_stopped` event **PR #49 @MZupa**
- Ability to debug simulations run through the `SoSID Viewer` with the PTSVD
  debugger (Closes #22) **PR #45 @skilkis**
- Ability to interact with other agents by running lambda functions and unbound
  methods using `Agent.run_on_other_agents` **PR #55 @skilkis**
- Unit-tests to check task population order and removal of tasks **PR #53
  @skilkis**
- Unit-tests to expose the infinite `Task` recursion bug **PR #39 @NabihNaeem
  @skilkis**

### Changed

- Minimum version of `flake8` to v3.8.0a2 to allow usage of the Python 3.8
  walrus operator **PR #45 @skilkis**
- `setup.py` to have `flake8` exclude the `.env` virtual environment folder
  to speed-up local linting (Closes #31) **PR #51 @skilkis**

### Fixed

- Infinite `Task` recursion bug when calling `TaskScheduler.set_active` by
  using a context-manager approach that allows intermediate calls to be made
  before callbacks are run (Closes #21) **PR #39 @NabihNaeem @skilkis**
- `SuppressionUAV` agents randomly flying to the top-left corner by correcting
  the Moore neighborhood used to track the firefront (Closes #21,
  #41) **PR #59 @NabihNaeem**
- Incorrect burnt-area calculation that did not previously account for
  suppressed yet burnt cells (Closes #58) **PR #61 @MZupa**

## [0.17.0] - 2020-04-28

### Added

- GitHub Actions CI/CD workflow with linting, style-checking, testing, and code
  coverage checking
- Extra `cicd` specifier for CI/CD pipeline to avoid installing the large
  `pyqt5-tools` and `jupyter` packages.

### Changed

- Code coverage report filename from `cov.xml` to `coverage.xml`

## [0.16.0] - 2020-04-21

### Added

- Section on coding best practices to [CONTRIBUTING]

### Changed

- Python version of project to 3.8 (Closes #23)
- `setup.cfg` with updated dependencies and metadata for the Python 3.8 upgrade

### Fixed

- Fix failing doc tests due to updated html_class in the new Sphinx version

## [0.15.0] - 2020-02-12

### Added

- `display` function to handle instantiation of the SoSID Viewer for any
  simulation object (Closes #6)
- `Deprecated` package to handle deprecated functions/classes with a decorator
- `main` file into `examples/wildfire` to easily launch the Wildfire example

### Changed

- `setup.cfg` to include `Deprecated` package

### Deprecated

- `gui.main` hard-coded function that would only launch the SoSID Viewer with
  the Wildfire example

### Removed

- Script for launching the Wildfire example from `examples/wildfire/simulation`

### Fixed

- Issue of SoSID Viewer not launching at first with IPython due to C++ object
  garbage collection (Closes #18)

### Known Issues

- `SuppressionUAV` agents randomly fly to the top-left corner of the mission
  area

## [0.14.0] - 2020-01-10

### Added

- Preliminary working logic for the `SuppressionUAV` which addresses the
  previous known issue regarding agents not functioning properly
- A modified `abc` module for defining abstract attributes that can be
  overridden by instance attributes.
- Benchmark of fire-modles using a Ryzen 3700x and GTX 2070 system
- Ability to continously cycle tasks when using the `autopopulate` option from
  `TaskScheduler`
- E203 to extend-ignore for [black] compatibility with whitespace around the
  slice operator
- Additional check in `pos_to_index` to correctly detect MESA position matrices

### Changed

- Threads per block to 32x32 from 16x16 for GTX2070 system
- Renamed position `pos` to index `idx` for signifying tha tan index should be
  passed to all derived `RasterizedShape` classes
- Reduced the number of active `SuppressionUAV` objects for easier debugging

### Fixed

- Incorrect task being run by `run_active` method in `TaskScheduler` class
- Endurance of `SuppressionUAV` not updating properly
- MESA ContinousSpace ufunc casting issue by instantiating initial positions as
  float values.
- Incorrect PIL sign convention being exposed in `RasterizedShape` to the user
  by replacing it with the Cellular Coordinate System (CCS)
- Incorrect bounds check on `RasterizedShape.nonzero` resulting in `ValueError`
- Incorrect distance travelled by a MovingAgent in a single simulation tick

### Known Issues

- NaN values can be output by the fire model for the `propagation_aspect`
  which leads the `SuppressionUAV` to raise errors.
- `SuppressionUAV` objects travel randomly to the top-left corner of the
  terrain even though they are not assigned to do so

## [0.13.0] - 2019-12-01

### Added

- Verification for CPU fire model ideal time step
- Ideal time step option to CPU fire model
- Plots comparing CPU and GPU fire-model burnt area for verification purpose
- UML class and activity diagrams for the SoSID Toolkit

### Changed

- Simple matplotlib visualization to use `COLOR_TABLE`
- Location of fire-model scripts to a `scripts/` directory

### Fixed

- Stochastic option in GPU preallocation of CA arrays

### Known Issues

- `SuppressionUAV` agents in the wildfire example are not functioning properly

## [0.12.0] - 2019-11-08

### Added

- Legacy fire models to `legacy` folder including the shared-memory aproning
  algorithm for the GPU fire-model
- Visualization for legacy fire-model
- `benchmark` module for CA fire-models

### Changed

- `setup.py` to ignore legacy code in `examples/wildfire/fire_model/legacy`
- Legacy fire-models to static time-step for benchmarking comparison
- Threads-per-block of GPU fire-model to 16x16 from 8x8

### Removed

- Dependency on `cupy` package for the GPU fire-model

### Known Issues

- `SuppressionUAV` agents in the wildfire example are not functioning properly

## [0.11.0] - 2019-11-04

## Added

- Classifiers and additional keywords (metadata) to `setup.cfg`
- `use_scm_version` to arguments in `setup.py`
- Information on how to perform an editable installation in [CONTRIBUTING]
- DLR MPL 2.0 license header to most source files

## Changed

- Setup configuration file with
- Numba minimum version requirement to version 0.46
- PyQt5 minimum version requirement due to `pyqt5-tools` package
- `paths.py` file to use `pkg_resources`
- Sphinx documentation version to use `get_distribution` from `pkg_resources`
- Import of generated terrain data to use `os.path` to make it more robust

## Removed

- `sosid` from `known_third_party` in coverage.py configuration
- Hardcoded version from Sphinx documentation

## Fixed

- Coverage report not including all packages by adding `__init__.py` files to
  wildfire example and ABM model
- Import of `Sequence` class from Numba which was causing pytest discovery to
  fail
- Declaration of `long_description` and `license` in `setup.cfg`
- Full-path issue in XML coverage report which caused the output filenames to
  be relative to the `examples` folder

## Known Issues

- `SuppressionUAV` agents in the wildfire example are not functioning properly

## [0.10.0] - 2019-11-03

*This is a backwards incompatible update, that adopts the `src/` layout*

### Added

- Autopopulation option to the Agent task system
- `typedef` file to contain all project type definitions
- Functions to transform between Cellular Coordinate System (CCS) and
  Global Coordinate System (GCS)
- Add utility to generate procedural (artificial) terrain
- Future improvements section to [CONTRIBUTING]
- `Environment` class housing both
- GUI Representation Protocol for rendering any simulation environment
  through adoption of `__gui_repr__`
- Sample procedurally generated terrain files in `wildfire/data/terrain`
- `Model` abstract base-class and implement Cellular Automata (CA) and Agent
  Based Model (ABM) with it
- `Simulation` class which acts as a threaded controller that can start, stop,
  and pause simulations
- Object-Oriented version of the CA fire-model that implements the
  `CellularAutomataModel` base class
- A ported version of the `cached_property` decorator from CPython 3.8
- Preliminary `SuppressionUAV` agent to the wildfire example

### Changed

- `core` folder name to `sosid` and adopted the `src/` layout in order to
  test the installed code and prevent package naming conflicts
- Fire-model combustibility to a uniform distribution
- Prototype GUI for viewing simulations to implement the GUI Representation
  Protocol
- `setup.py` and `setup.cfg` to comply with the `src/` layout

### Known Issues

- `SuppressionUAV` agents in the wildfire example are not functioning properly
- Test discovery fails due to relocation of Numba's `Sequence` change

## [0.9.0] - 2019-10-29

### Added

- Ability to influence Cellular Automata (CA) models from a continous space
  through use of `PIL` library rasterization, along with a few primitive
  shapes. These are accesible through the `raster` module
- Agent task system supporting task queuing and prioritization
- Support for Digital Elevation Model (DEM) terrains
- Unit-tests for the new `raster`, `task`, and `terrain` modules
- Git LFS support for Numpy binary array files

### Changed

- `reverse_2d` function into `reverse_angle` for clarity
- `COLOR_TABLE` to use more realistic colors
- Location of decorators into core directory
- Sphinx documentation appearance of decorators and re-implemented footer
  border
- Expected test results to module-level constant style for readability
- Order of iteration indices in GPU fire-model for performance. This better
  adheres to the x to the right, y-down convention of CUDA.

### Removed

- End of Line (EOL) option from `.editorconfig` in favor of handling them
  with `.gitattributes`

### Fixed

- `ScenarioTestSuite` pytest caching error on unhashable types by using the
  `Scenario` class for hashing purposes

### Known Issues

- Test discovery fails due to relocation of Numba's `Sequence` change

## [0.8.0] - 2019-10-11

### Added

- Prototype PyQt5 based simulation visualization
- `COLOR_TABLE` for mapping fire-states to colors
- `annotation` module into Sphinx documentation
- `ScenarioTestSuite` class into test snippets to simplify parameterization of
  classes with initialization arguments.

### Changed

- Location of `sum_neighbors` JIT function into the `common.py` module
- Location of `gui` folder into `core` directory
- `AnnotationInterpreter` class into a function to simplify its API
- Unit-tests for Cellular Automata neighborhoods

### Fixed

- `flake8` F401 error code applying to the entire code-base and not just to the
  `tests/` directory
- Faulty cell count in NeumannNeighborhood

### Known Issues

- Test discovery fails due to relocation of Numba's `Sequence` change

## [0.7.0] - 2019-10-04

### Added

- `reverse_2d` JIT function to reverse angles
- CUDA stream synchronization for benchmarking. Previously the kernel would
  launch asynchroniously causing only the kernel launch time to be measured.

### Changed

- `wind_aspect` direction changed from the Rui, 2018 convention to
  meteorological convention (where the wind is coming from)
- Location of `array_ops` module into `model.ca.jit_funcs` package

### Fixed

- `None` not being properly converted from an annotation into Numba signature
- Docstring of `compile_jit` fixture

### Known Issues

- Test discovery fails due to relocation of Numba's `Sequence` change

## [0.6.0] - 2019-10-01

### Added

- [CONTRIBUTING] file to help future developers
- Decorator to inject attributes into objects
- `.editorconfig` to control coding styles across multiple systems
- Style-checking of import statements with `flake8-isort`
- Code badges to `README.md`
- Metadata to `setup.cfg`

### Changed

- Firefighting Sphinx documentation by adding CUDA implementation paragraph
- Unified CPU and GPU fire-model visualization in one file
- Refractored preallocation of GPU and CPU arrays to seperate functions
- Simplified iteration bounds of CPU fire-model by taking into account the
  radius of the moore neighborhood
- Flattened test directory to a maximum depth of 3 folders

### Removed

- `requirements.txt` file that is no longer needed
- Aproning algorithm for GPU fire-model
- Redundant tests and old files
- Old location of `wildfire.py` module

### Fixed

- Markdown syntax to include empty lines
- Minor errors in docstrings of `states.py`

### Known Issues

- Test discovery fails due to relocation of Numba's `Sequence` change

## [0.5.0] - 2019-09-27

### Added

- Arguments for terrain slope and terrain aspect
- Utility to measure the size of CA data arrays
- `FAST_MATH_FLAGS` option for GPU fire-model
- Tests for the `geom2d` module
- Shared memory aproning algorithm for GPU fire-model

### Changed

- Reduced the number of redundant operations in the `aspect_2d` JIT function

### Removed

- Deprecated `fast_math.py` module
- Old test `test_propagate.py` module

### Fixed

- Intersphinx references in `conf.py`
- All mentions of Ahead of Time (AOT) compilation which was misinterpreted.
  In reality the testing process used eager Just in Time (JIT) compilation
  instead

## [0.4.0] - 2019-09-25

### Added

- Ability to transform type-hint annotations into Numba signatures for
  simplifying JIT compilation
- Collection of re-usable tests with `snippets` and `conftest` modules
- Unit-tests for common fire-model JIT functions
- `FAST_MATH_FLAGS` option for JIT functions to provide LLVM compiler flags

### Changed

- Location of Numba and Sphinx test helpers into `util` folder
- Location of `fast_math` module

### Fixed

- Incorrect slope coefficient value caused by input in degrees
- Incorrect hill direction slope coefficient
- Return of a value when expecting NaN
- Missing Annotation in `aspect_2d` JIT function

## [0.3.0] - 2019-09-18

### Added

- Sphinx support for `@cuda.jit` decorated functions
- Coverage report with coverage.py
- Additional `flake8` configuration options in `setup.cfg`

### Fixed

- Incorrect type declaration in `numba_patch.py`
- `adopts` typo to `addopts` in `setup.cfg`

## [0.2.0] - 2019-09-12

### Added

- Patch to make Numba JIT decorated functions compatible with Sphinx
- More .rst files for documentation
- Small utility to calculate fire model array size
- Initial GPU fire model
- Initial testing for JIT-compiled functions

### Changed

- Fire model directory to have seperate `cpu` and `gpu` folders

## [0.1.2] - 2019-09-11

### Added

- Dark header theme for Sphinx documentation
- Funnel/filter logo for Sphinx Documentatin
- Preliminary pytest config within `setup.cfg`
- Configuration for [black] code formatter with a `pyproject.toml` file
- Google docstring enforcement with `flake8-docstrings` package

### Changed

- Autodoc formatting with tabs for readability
- Unified simulation set-up for cpu_padded and cpu_inplace fire models
- Fire states to be captialized in order to adhere with PEP-8
- `.flake8` file contents moved into `setup.cfg`

### Fixed

- In-place summation in spread-rate array causing faulty values for fire-model
- Spread rate being calculated for cells other than those at FULL_BURNING which
  caused different behavior than the model of Rui, 2018
- Behavior of Sphinx HTML documentation navbar on right-click when out of focus

## [0.1.1] - 2019-09-03

### Added

- Core directory containing (CA) helper modules
- Working Parallel CPU based Fire-Model
- Simple Matplotlib-based visualization of Fire-Model
- Ability to add PlantUML diagrams to Sphinx documentation
- Intersphinx mapping to link to Numpy and Numba types
- Sphinx Auto inheritence diagrams using Graphviz
- Exprimental abstractions for defining simulation variables & parameters

### Changed

- Structure and filenames of fire-propagation model

### Known Issues

- Tests discovery fails due to improper imports

## [0.1.0] - 2019-08-21

### Added

- Unit Tests Directory using pytest
- Added Documentation (Set-up Sphinx w/ bootstrap theme, & Added ADR)

<!-- Un-wrapped Text Below for References, Links, Images, etc. -->
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[black]: https://black.readthedocs.io/en/stable/
[Contributing]: CONTRIBUTING.md
