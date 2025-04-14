# Contributing to the SoSID Toolkit (A Guide for Developers)

## Installation

### 1. Cloning the Repository

To grab the latest version of the SoSID toolkit run the following command:

```cmd
git clone https://github.com/pandaworksSOS/sosid_toolkit.git
```

### 2. Creating and Activating a Virtual Environment

To ensure that the installation does not affect other projects, it is wise to
create a new local virtual environment. For this we will use the `venv` module,
however you are free to use `conda` if you wish.

```cmd
py -3 -m venv .env
```

Once this command executes, the new environment must be activated as follows:

#### Windows

```cmd
".env/Scripts/activate"
```

_Note: the quotation marks are important!_

#### Linux

```bash
source .env/Scripts/activate
```

### 3. Installing an Editable Version of the SoSID Toolkit

Next, since this project uses the `src/` layout an editable version of the code
needs to be installed. This is accomplished by running the following command:

```cmd
pip install -e .[dev]
```

Note that the `[dev]` specifier here installs some additional requirements
necessary for proper development such as a syntax checker and a formatter.
The following specifiers can be added to the installation command:
- `[dev]`: helps with code development and is required for clean code programming
- `[docs]`: used when sphinx documentation is applied
- `[cicd]`:  similar to `dev` but does not apply for development purposes and commits
- `[geo]`: used for terrain generation in wildfire and UAM graph generation
- `[graph_viz]`: used for UAM graph visualization- NOT REQUIRED FOR WILDFIRE

For example if one wishes to only download the specifiers for wildfire simulation:

```cmd
pip install -e .[dev,docs,cicd,geo]
```

### 4. Running the Wildfire Example

Currently the terrain data which is stored as Numpy .npy files is registered
in git with Large-File-Support (LFS), we need to first make sure that LFS is
installed:

```cmd
git lfs install
```

### 5. Install pre-commit hooks

Lastly, the project adopts certain standards for the code and commit messages.
In order to simplify compliance with these standards through automated checks,
we need to setup `pre-commit`:

```cmd
pre-commit install
```

## Cross-Platform / Cross-IDE Development

To ensure that the code indentation, spacing, and line-endings are normalized
across all platforms and IDE's an `.editorconfig` file is used. Make sure that
your editor supports this configuration file by checking the [editorconfig]
website. Depending on your editor you may need to install a plug-in/extension
which is the case currently for [Visual Studio Code].

## Best-Practices

### Code Formatting

To preserve a uniform look across the entire project [ruff]. This is especially
important when working with multiple developers as everyone usually has their
own formatting tastes. However, as with all joint endeavors the way forward is
through compromise...or in this case an auto-formatter with a heavy hand. All of
the required packages and their respect extensions will be automatically
installed if the `[dev]` specifier is added when installing the SoSID Toolkit.

`ruff` has been adopted at a later stage in the project and as it is preferred
to not change the entire codebase at once, it is recommended to run `ruff` only
on the git diff. This will allow for a gradual transition to the new formatting
rules. To run `ruff` on the git diff, the following command can be used (where
`--help` can be used to see the available options):
```cmd
py scripts/ruff_check.py
py scripts/ruff_format.py
```

### Git Commits

To have a rich history to the commits that are easily understood by all devs,
the following best-practices that are widely accepted/used in open-source
projects are adopted.

#### The seven rules of a great Git commit message

1. Separate subject from body with a blank line
2. Limit the subject line to 50 characters
3. Capitalize the subject line
4. Do not end the subject line with a period
5. Use the imperative mood in the subject line
6. Wrap the body at 72 characters
7. Use the body to explain what and why vs. how

The most important rule is to have a descriptive subject line that is written
in the imperative mood such that you can read the history as `This commit
will...{add subject line here}`. Therefore, past tense needs to be avoided and
subject lines starting with `Changed, Added, Modified` should be replaced with
`Change, Add, Modify`. Thus, an example commit would be as follows:

```git
Add main.py to easily run the bot

This commit adds a main.py module that makes it a lot easier to initialize and
run the bot. Before, the bot would need to be imported from multiple files
which made it difficult for hosting the bot on the cloud.

Therefore, this module takes care of these imports and runs the bot using the
bot.run command.

Closes: #1
```

More information on these best-practices can be obtained from: [git-commits]

### Git Branches and Releases

The repository contains one main branch: `master`. All the developments should
take place on a branch created from `master` with a name following the
convention: `classifier/description`. The `classifier` can be one of the
following: `feature`, `codequality`, `bugfix`, and `demo`. The `description`
should be a short descriptor for the developments taking place in that branch.
These development branches are then merged into the `master` in short intervals.
Periodically, a `release` shall be made from the stable `master` branch.

The following steps should be followed during the creation of a new release.

1. Make a `release` branch from `master`
2. Test the correct function of the code-base by running the test-suite on
   `release` and manual tests of the examples. Apply at hotfixes necessary
3. Update [CHANGELOG.md] with information pertaining to the new release
4. Create a commit titled `Update CHANGELOG for Release vX.Y.Z` and tag it with
   a new git tag labeled with `vX.Y.Z`. This is required for `setup.py` to pick
   up on the new release version when installing the bot.
5. Merge `release` branch back into `master`.
6. Form a GitHub release with the commit tag created in **Step 3** and copy
   the latest relevent changes from [CHANGELOG.md] into the release body.

_In the future, this process can be automated with CI/CD tools, but due to the
short time-frame of the initial release, automation will be dealt with later._

## Documentation

Python objects can be cross-referenced in Sphinx documentation utilizing
[SphinxDomains], [sphobinv]. As an example, finding the functools module
from the standard library can be done as follows:

```cmd
sphobjinv suggest [URLto objects.inv] [desired object] -u
```

## Future Improvements (Help Needed!)

-   Introduction of a logger for the ABM task system that is able to
    that stores a history of all tasks (actions) performed by the agent.
-   Automatic generation of [PlantUML] activity diagram for an Agent by using
    the Python [AST] (Abstract Syntax Tree) package for code introspection.
    Such a feature would make use of the ABM task system.
-   Add testing for build process of Sphinx documentation
-   Test [VisPy] and see if it has improvements over the current Qt and pyqtgraph
    visualization.
-   Relaxing version specifiers of dependencies in [setup.cfg] after testing with
    tox.
-   Look into implementing a `__model_interface__` similar to the functionality
    of the Numpy `__array_interface__` to enable coordinate transforms between
    arbitrary coordinates of different Models.

<!-- Un-wrapped URL Links -->

[SphinxDomains]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
[sphobjinv]: https://github.com/bskinn/sphobjinv
[PlantUML]: http://plantuml.com/
[AST]: https://docs.python.org/3/library/ast.html
[VisPy]: http://vispy.org/index.html
[setup.cfg]: setup.cfg
[vegindex]: https://pypi.org/project/vegindex/
[editorconfig]: https://editorconfig.org/
[Visual Studio Code]: https://code.visualstudio.com/
[git-commits]: https://chris.beams.io/posts/git-commit/
[flake8]: https://flake8.pycqa.org/en/latest/
[black]: https://black.readthedocs.io/en/stable/
[CHANGELOG.md]: CHANGELOG.md
