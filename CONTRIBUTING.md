## Contributing to PDE ContRoL Gym
Contributions from the open-source community are certainly welcomed and encouraged. 

### Contributing Guidelines
Please follow the series of pipelines to submit your contributions below. Remember, contributions are not just in the form of code, but also in forms of issues where items like Bug Reports are extremely valuable. For all types of contributions, please follow the set series of steps to ensure that your contribution makes production.

- First, check that your test/feature/bugfix already is available and discussed in the issues tab as a valuable addition to the codebase.
  - If your contribution is not yet listed on the issues tab, please create a new issue with the appropriate tag and @lukebhan on the issue as well. In many cases, a quick response will be given and a resulting discussion may then occur about the fix, its implementation, and potentiall testing required. 
- Next, please create a pull request following the **Style Guidelines** listed below. It is required that your pull request is reviewed by at least one Admin before merging. 
- Third, please then squash your pull request, double check the style guidelines and you are ready to merge your contribution! Thank you for your efforts. 

#### Style Guidelines
For style guidelines, we use <a href=https://github.com/psf/black>black</a> as our formatter. Please ensure every file (new and old) is formatted under the default Black code style. (This is technically a PEP8 compliant formatter - so if you prefer to use just PEP8, explicily say so in your pull request and we can discuss)

#### Documentation
Contributions to the documentation are always welcome. The documentation is built on Sphinx using autobuild. Please ensure that your contribution to the documentation is not insignificant (ie a small one line typo is not something worth spending time on, but explaining all the parameters of a new feature certainly is).

To set up the documentation, create an environment and install the packages in `docs/requirements.txt`. Installing the gym itself as well is what lets `autodoc` pull in the environment and reward docstrings.

```
python3 -m venv .venv-docs
source .venv-docs/bin/activate
pip install -r docs/requirements.txt
pip install -e .
```

A one-off build, run from the `docs/` folder, writes HTML to `docs/build/html`:

```
cd docs
make html
```

Open `docs/build/html/index.html` in a browser to read it. While editing, `sphinx-autobuild` rebuilds on save and serves the result at http://127.0.0.1:8000:

```
cd docs
sphinx-autobuild source build/html
```

`make clean` removes the build directory if a stale build needs clearing. Note that the build currently emits a number of pre-existing warnings from the BrainTumor, Neuron and Traffic pages, so check that your contribution has not *added* warnings rather than expecting a silent build.

