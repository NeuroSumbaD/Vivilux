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

@pytest.mark.benchmark(group="ra25_init")
def test_ra25_init(benchmark):
    '''
    Benchmark ra25 initialization performance
    '''
    benchmark(lambda: ra25.ra25_init())
    
@pytest.mark.benchmark(group="ra25_num_hidden_layers")
def test_ra25_hidden_layer_2(benchmark):
    '''
    Benchmark ra25 training completion for 2 hidden layers
    '''
    leabraNet2 = ra25.ra25_init(numHiddenLayers=2, errorThreshold=0)
    benchmark(lambda: leabraNet2.Learn(input=ra25.inputs, target=ra25.targets,
                                numEpochs=5,
                                reset=False,
                                shuffle = False,
                                EvaluateFirst=False
                                ))

@pytest.mark.benchmark(group="ra25_num_hidden_layers")
def test_ra25_hidden_layer_3(benchmark):
    '''
    Benchmark ra25 training completion for 3 hidden layers
    '''
    leabraNet3 = ra25.ra25_init(numHiddenLayers=3, errorThreshold=0)
    benchmark(lambda: leabraNet3.Learn(input=ra25.inputs, target=ra25.targets,
                                numEpochs=5,
                                reset=False,
                                shuffle = False,
                                EvaluateFirst=False
                                ))

@pytest.mark.benchmark(group="ra25_num_hidden_layers")
def test_ra25_hidden_layer_4(benchmark):
    '''
    Benchmark ra25 training completion for 4 hidden layers
    '''
    leabraNet4 = ra25.ra25_init(numHiddenLayers=4, errorThreshold=0)
    benchmark(lambda: leabraNet4.Learn(input=ra25.inputs, target=ra25.targets,
                                numEpochs=5,
                                reset=False,
                                shuffle = False,
                                EvaluateFirst=False
                                ))
    
THRESHOLD = 0.005
MAX_EPOCHS = 15

@pytest.mark.ra25_train
def test_ra25_train():
    '''
    Test ra25 training convergence. The test will fail if the error THRESHOLD is not reached in MAX_EPOCHS.
    '''

    leabraNet = ra25.ra25_init(errorThreshold=THRESHOLD)
    def num_epochs():
        result = leabraNet.Learn(input=ra25.inputs, target=ra25.targets,
                                numEpochs=MAX_EPOCHS,
                                reset=False,
                                shuffle = False,
                                EvaluateFirst=False
                                )
        numEpochs = len(result[next(iter(result))])
        logger.info(f"test_ra25_train: Number of epochs taken: {numEpochs}")
        logger.info(f"test_ra25_train: Error: {result['AvgSSE'][-1]}")
        return numEpochs
    
    epochs_taken = num_epochs()
    assert epochs_taken < MAX_EPOCHS, f"Training did not reach the error threshold {THRESHOLD} in {MAX_EPOCHS} epochs"
