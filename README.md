# MEng Project

## About the Project

Currently, motif-based DNA storage is a cheap DNA storage option where only short DNA sequences are constructed and then assembled. However, due to biological and technological constraints involved in storing data in molecules, many encodings of arbitrary data to DNA end up being extremely error-prone, making DNA storage not a reliable storage option. Having to find an encoding design which takes into account those constraints as well as encodes a given data can be time extensive and challenging.

So to facilitate the search for a motif-based encoding design which conforms to a set of biological and technological constraints, we implemented:

1. A motif generation tool using Markov Chains which outputs a set of keys and payloads given a set of constraints.

The tool can be found on: https://motif-generation-tool.herokuapp.com

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

To run the Motif Generation Tool, run the following command from the root directory:
```bash
python -m motif_generation_tool.key_payload_builder run
```

### Tests

To run the tests, first go to the unit_tests directory (motif_generation_tool/unit_tests).

Run tests using the following command line: 
```bash
pytest filename.py
```
For example: 
```bash
pytest hairpin_tests.py
```
