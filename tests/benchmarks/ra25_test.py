import pytest

import sys
import pathlib
import logging

#TODO: REMOVE THIS LINE, USED FOR SUPPRESSING UNWANTED WARNINGS IN DEBUGGING
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Equivalence directory to the path
equivalence_dir = pathlib.Path(__file__).parent.parent / 'Equivalence'
sys.path.insert(0, str(equivalence_dir))

import ra25

@pytest.mark.benchmark(group="init")
def test_ra25_init(benchmark):
    '''
    Benchmark ra25 initialization performance
    '''
    benchmark(lambda: ra25.ra25_init())

THRESHOLD = 0.005
MAX_EPOCHS = 15

def test_ra25_train():
    '''
    Test ra25 training convergence. The test will fail if the error THRESHOLD is not reached in MAX_EPOCHS.
    '''

    leabraNet = ra25.ra25_init(error_threshold=THRESHOLD)
    def num_epochs():
        result = leabraNet.Learn(input=ra25.inputs, target=ra25.targets,
                                numEpochs=MAX_EPOCHS,
                                reset=False,
                                shuffle = False,
                                EvaluateFirst=False
                                )
        numEpochs = len(result[next(iter(result))])
        logger.info(f"Number of epochs taken: {numEpochs}")
        logger.info(f"Error: {result['AvgSSE'][-1]}")
        return numEpochs
    
    epochs_taken = num_epochs()
    assert epochs_taken < MAX_EPOCHS, f"Training did not reach the error threshold {THRESHOLD} in {MAX_EPOCHS} epochs"
