# Visualization (GUI) Framework

* Status: Accepted
* Deciders: San Kilkis
* Date: 2019-10-09

## Context and Problem Statement

An important feature to being able to set-up the simulation environment is to
be able to visualize the interaction of agents with their environment.
Therefore, although a full-fledged GUI is not currently in question, it is
important select a proper visualizaiton/GUI framework that is extensible and
allows for high-performance rendering of the simulation scene. Furthermore,
all of the considered options are cross-platform. As such, this is not
listed as a positive outcome as it is taken for granted.

## Decision Drivers

* Performance
* Simplicity
* Extensibility

## Considered Options

* [PyQt5]
* [wxPython]
* [Kivy]
* [Tkinter]

## Decision Outcome

Chosen option: [PyQT5]. This means that licensing might become an issue in the
future but since PyQT5 has great documentation, examples, and good support
through its vast online community development will be much easier. Also, the
availability of [PyQtGraph], translates to the quickest way to achieve high
performance rendering of arrays, which is necessary for displaying the
high cell counts present in the Cellular-Automata (CA) fire model. If an
alternative is required due to licensing, the next-best option for the SOSID
toolkit is [Kivy].

## Pros and Cons of the Options

### [PyQt5]

* Good, because it is a professional library that has many modules
* Good, because it comes bundled with lots of examples
* Good, because it has excellent support and documentation
* Good, because it has Qt Creator for fast GUI prototyping
* Good, because [PyQtGraph] enables fast rendering of Numpy arrays
* Good, because it comes with support for .svg and path painting
* Bad, because Qt has restrictive licensing
* Bad, because the bulk of the library means there is a lot to learn

### [wxPython]

* Good, because it has unresctrictive licensing
* Good, because [wxGlade] provides an interactive GUI Builder
* Good, because it is a popular library with examples and documentation
* Bad, because it does not have good support for drawing paths, objects

### [Kivy]

* Good, because it has unrestrictive licensing
* Good, because it allows for GPU acceleration
* Good, because it has good support for drawing paths and objects
* Good, because it comes with great documentation
* Good, because it comes wiht a GUI designer
* Bad, because there is no simple way to implement a color map for Numpy arrays

### [Tkinter]

* Good, because it is very simple
* Good, because it is built-into Python
* Good, because it is suprisingly powerful for its small footprint
* Bad, because it is not very extensible and lacks lots of widgets
* Bad, because the documentation is not great although there are a lot of
  examples

<!-- Unwrapped URLs -->
[PyQt5]: https://pypi.org/project/PyQt5/
[wxPython]: https://pypi.org/project/wxPython/
[Kivy]: https://pypi.org/project/Kivy/
[Tkinter]: https://docs.python.org/3/library/tk.html
[PyQTGraph]: http://www.pyqtgraph.org/
[wxGlade]: http://wxglade.sourceforge.net/
