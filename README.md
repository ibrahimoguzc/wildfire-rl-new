# System of Systems Inverse Design (SoSID) Toolkit

[![Supported Python Versions][python_badge]](https://www.python.org/)
[![License: MPL 2.0][mpl_badge]](https://opensource.org/licenses/MPL-2.0)
[![Code Style: Black][black_badge]](https://github.com/ambv/black)
[![CICD: GitHub Actions][build_status]](https://github.com/pandaworksSOS/sosid_toolkit/actions)

Welcome! Utilizing System of Systems Inverse Design paradigm is an exciting new
development for aircraft designers. SoSID has the potential to reveal the
relation between system-level (aircraft) Metrics of Performance (MoPs) and
mission-level Metrics of Effectiveness (MoEs).

Just as MDO quantifies the hard to comprehend insight into how discipline-level
decisions impact the performance of an aircraft, SoSID enables an aircraft
designer to quantify how top-level requirements affect the operational
environment. Therefore, more cost effective solutions to a given problem can be
designed that take into account the limitations of other systems.

This repository aims to accelerate the SoSID analysis of future aircraft by
providing useful abstractions for use in defining ABM simulations as well as
linking them to other modelling techniques. A sample use-case of an inverse
design on a wildfire suppression UAV is provided. The goal is to constantly
test lower-level concretions, such as a fire propagation model and operational
logic of firefighting agents, to gain insight into the usefulness of the
high-level abstractions provided in the toolkit.

<!-- Un-wrapped URL's below (Mostly for Badges) -->
[python_badge]: https://img.shields.io/badge/python-3.10%20|%203.11-blue.svg
[mpl_badge]: https://img.shields.io/badge/license-MPL%202.0-brightgreen.svg
[black_badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[build_status]: https://github.com/skilkis/sosid_toolkit/workflows/build/badge.svg
