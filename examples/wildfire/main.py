"""Script used to run the Wildfire example and launch the SoSID Viewer.

Arguments can be passed to :py:class:`WildfireParameters` in order to
replace default values of the simulation::

    params = WildfireParameters(temperature=20, flight_velocity=25)

"""

from pathlib import Path

from pyinstrument import Profiler

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation
from sosid.gui.main import display
from sosid.output import OutputFormat
from sosid.util.general_funcs import combine_parameters

INPUT_FILE = "Salamis.json"
OVERWRITES = None
RUN_PROFILER = False

OUTPUT_DIR = SCENARIOS_DIR / "outputs"
input_filepath = SCENARIOS_DIR / "inputs" / INPUT_FILE
if OVERWRITES:
    overwrite_filepath = Path(input_filepath).parent / OVERWRITES
    params = combine_parameters(input_filepath, overwrite_filepath)
    parameters = WildfireParameters.model_validate(params)
else:
    with input_filepath.open() as f:
        parameters = WildfireParameters.model_validate_json(f.read())

sim = WildfireSimulation(
    parameters=parameters,
    seed=0,
    context=Profiler(interval=0.001) if RUN_PROFILER else None,
)

if RUN_PROFILER:
    sim.start()
    sim.join()
    sim.context.open_in_browser()
elif parameters.run_headless:
    print("Running Headless")
    sim.start()
    sim.is_stopped.wait()
else:
    display(sim)

# Call the output method if the simulation is terminated.
if sim.is_stopped.is_set():
    sim.write_output_data(
        output_holder := sim.get_output_data(),
        OUTPUT_DIR / (input_filepath.stem + "_out"),
        OutputFormat.JSON,
    )
