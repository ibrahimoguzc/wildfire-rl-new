# Changing Main Package Name Within Repository

* Status: Accepted
* Deciders: San Kilkis
* Date: 2019-09-12

## Context and Problem Statement

The current folder containing the main source code within the repository
`sosid_toolkit\code` does not adhere to the src-layout or adhoc-layout
`package_name\package_name` style of popular Python packages such as Numpy,
Numba, Sphinx to list a few. Therefore, the problem is to decide on a
convention for this source code folder.

## Decision Drivers

* Adhering to best-package
* Packaging

## Considered Options

* Keeping it the way it (`sosid_toolkit\core`)
* Changing to a src-layout (`sosid_toolkit\src\sosid_toolkit`)
* Changing to an adhoc-layout (`sosid_toolkit\sosid_toolkit`)

## Decision Outcome

Chosen option: Changing to a src-layout (`sosid_toolkit\src\sosid_toolkit`).
Due to the increased verbosity, the name of the package should be reconsidered
and preferably changed to one that does not involve special characters.
Although, the src-layout is still relatively new, it was discussed in
[PyCon 2019] and the consensus was that it has benefits for packages with many
modules.

### Positive Consequences

* Adopts an emergent convention that has the potential to the accepted standard
* Prevents source code from being mixed-up with installed code
* Developers are forced to test the installed code (Discussed by: Ionel
  Cristian Mărieș in [Packaging a Python Library])

### Negative Consequences

* Importing modules with absolute imports becomes more vebose
* To start developing, one has to install the package with the pip -e option

## Pros and Cons of the Options

### Keeping it the way it (`sosid_toolkit\core`)

* Good, because no refractoring/renaming required
* Bad, because it does not adhere to any particular standard or best-practice

### Changing to an src-layout (`sosid_toolkit\src\sosid_toolkit`)

* Good, because it forces developers to test the installed code
* Good, because it tests out a new convention that isn't currently used in DLR
* Bad, since it is still not widely adopted and there are fewer examples

### Changing to an adhoc-layout (`sosid_toolkit\sosid_toolkit`)

* Good, because it follows best-practice established by popular packages
* Good, because it was recommended by Jan-Niclas Walther
* Bad, because source code can become mixed up with installed code (mentioned by [pytest-cov])

<!-- Long Un-wrapped URLS for Links -->
[PyCon 2019]: https://docs.google.com/document/d/1Wz2-ECkicJgAmQDxMFivWmU2ZunKvPZ2UfQ59zDGj7g/edit#heading=h.2cgqnlxl8y3e
[pytest-cov]: https://github.com/pytest-dev/pytest-cov/tree/master/examples
[Packaging a Python Library]: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
