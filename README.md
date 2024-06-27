# MEng Project

## About the Project

Currently, motif-based DNA storage is a cheap DNA storage option where only short DNA sequences are constructed and then assembled. However, due to biological and technological constraints involved in storing data in molecules, many encodings of arbitrary data to DNA end up being extremely error-prone, making DNA storage not a reliable storage option. Having to find an encoding design which takes into account those constraints as well as encodes a given data can be time extensive and challenging.

So to facilitate the search for a motif-based encoding design which conforms to a set of biological and technological constraints, we implemented:

1. A motif generation tool using Markov Chains which outputs a set of keys and payloads given a set of constraints.

The tool can be found on: https://ssb5018.pythonanywhere.com

## Getting Started

### Prerequisites
* Python 3.8 or above.

### Installations
```bash
pip install -r /path/to/requirements.txt
```
To install asyncio, run the following command:
```bash
pip install asyncio
```
## Usage
### Motif Generation Tool

To get a list of keys and set of payloads conforming to the given constraints, use the Motif Generation Tool. 

The constraints can be inputted directly in the main function of the motif_generation_tool/key_payload_builder.py file.

To run the Motif Generation Tool, run the following command from inside the motif_generation_tool directory:
```bash
python -m key_payload_builder run
```

#### Hyperparameter Tuning

To run the hyperparameter tuning, run the following command from inside the motif_generation_tool directory:
```bash
python -m hyperparameters.hyperparameter_tuning run
```

The hypTun.txt file inside of the hyperparameters folder holds the output of the first round of hyperparameter tuning. It is the result of running the hyperparameter_tuning.py file with shape values [10, 20, 30, 40, 50], and weight values set to 1.

The hypTun2.txt file inside of the hyperparameters folder holds the output of the second round of hyperparameter tuning. It is the result of the hyperparameter_tuning.py file with the following input for shape values (weight values are set to 1):

```bash
shape_values = {'hom': [50, 55, 60, 65, 70], 
                'hairpin': [2, 4, 6, 8, 10], 
                'similarity': [50, 55, 60, 65, 70], 
                'gcContent': [7, 10, 13, 16, 19], 
                'noKeyInPayload': [15, 20, 35, 40, 45]
                }
```

To view the results in form of a heatmap, run the dataExtract() function inside of the hyperparameter_tuning.py file.
The results of the first and second round of hyperparamter tuning can be found inside of the figures folder (motif_generation_tool/hyperparameters/figures).

### Tests

To run the tests, first go to the root directory.

Run tests using the following command line: 
```bash
python3 -m pytest unit_tests/[insert file name].py
```
For example: 
```bash
python3 -m pytest unit_tests/hairpin_tests.py
```
