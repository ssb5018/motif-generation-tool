from constraints.constraints import Constraints
from .hyperparameters import Hyperparameters
from key_payload_builder import KeyPayloadBuilder

import numpy as np
import time
import os

class HyperparameterTuning:
    def __init__(self, number_of_runs=100):
        self.number_of_runs = number_of_runs

    async def build_keys_and_payloads(self, constraints, hyperparameters, with_constraints):
        keyPayloadBuilder = KeyPayloadBuilder(constraints, hyperparameters)
        keys, payloads = await keyPayloadBuilder.build_keys_and_payloads(with_constraints)
        return keys, payloads
    
    async def round_validation(self, constraints, hyperparameters, with_constraints):
        keys, payloads = await self.build_keys_and_payloads(constraints, hyperparameters,\
                                                            with_constraints)
        if not (keys and payloads):
            return False
        return True

    async def round(self, constraints, hyperparameters, with_constraints):
        totalSuccesses = 0
        for i in range(self.number_of_runs):
            valid = await self.round_validation(constraints, hyperparameters, with_constraints)
            totalSuccesses += 1 if valid else 0
        return totalSuccesses
    
    async def grid_search(self, constraints, shape_values, weight_values, with_constraints):
        """This function takes a set of possible hyperparameter values and tries out all
        of their combinations. It stores the results in the file hypTun.txt.
        
        Parameters
        ----------
        shape_values: dict str: list int/float
            Dictionary of constraints to the values we want to try the shape hyperparameters 
            corresponding to each constraint for. Constraints are 'hairpin', 'hom', 'similarity',
            'gcContent'.
        weight_values: dict str: list int/float
            Dictionary of constraints to the values we want to try the weight hyperparameters 
            corresponding to each constraint for. Constraints are 'hairpin', 'hom', 'similarity',
            'gcContent'.
        with_constraints: set of str
            Set of strings containing a selection of the following constraints: 
            'hairpin', 'hom', 'gcContent'. Those will be the constraints that the
            palyoads and keys will have to conform to.
        """

        hom_shapes = [1] if not 'hom' in shape_values else shape_values['hom']
        hairpin_shapes = [1] if not 'hairpin' in shape_values else shape_values['hairpin']
        similarity_shapes = [1] if not 'similarity' in shape_values else shape_values['similarity']
        no_key_in_payload_shapes = [1] if not 'noKeyInPayload' in shape_values else shape_values['noKeyInPayload']
        gc_shapes = [1] if not 'gcContent' in shape_values else shape_values['gcContent']

        hom_weights = [1] if not 'hom' in weight_values else weight_values['hom']
        hairpin_weights = [1] if not 'hairpin' in weight_values else weight_values['hairpin']
        similarity_weights = [1] if not 'similarity' in weight_values \
                                 else weight_values['similarity']
        no_key_in_payload_weights = [1] if not 'noKeyInPayload' in weight_values \
                                 else weight_values['noKeyInPayload']
        gc_weights = [1] if not 'gcContent' in weight_values else weight_values['gcContent']

        for hom_shape in hom_shapes:
            for hom_weight in hom_weights:
                for hairpin_shape in hairpin_shapes:
                    for hairpin_weight in hairpin_weights:
                        for similarity_shape in similarity_shapes:
                            for similarity_weight in similarity_weights:
                                for no_key_in_payload_shape in no_key_in_payload_shapes:
                                    for no_key_in_payload_weight in no_key_in_payload_weights:
                                        for gc_shape in gc_shapes:
                                            for gc_weight in gc_weights:
                                                shapes = {'hom': hom_shape, 
                                                        'hairpin': hairpin_shape, 
                                                        'similarity': similarity_shape,
                                                        'noKeyInPayload': no_key_in_payload_shape,
                                                        'gcContent': gc_shape 
                                                        }
                                                weights = {'hom': hom_weight, 
                                                        'hairpin': hairpin_weight, 
                                                        'similarity': similarity_weight,
                                                        'noKeyInPayload': no_key_in_payload_weight,
                                                        'gcContent': gc_weight 
                                                        }
                                                hyp = Hyperparameters(shapes, weights)
                                                start = time.time()
                                                total_successes = await self.round(constraints, hyp, \
                                                                                    with_constraints)
                                                end = time.time()
                                                print('time: ', end - start)
                                                print('total successes: ', total_successes)
                                                w = 'weights: ' + str(weights) + 'shapes: ' + str(shapes) + \
                                                    'totalSuccesses: ' + str(total_successes) + '\n'
                                                f = open(os.path.join(os.path.dirname(__file__), "hypTun.txt"), "a")
                                                f.write(w)
                                                f.close()


async def main():
    payload_size = 60
    payload_num = 15
    max_hom = 5
    max_hairpin = 1
    loop_size_min = 6
    loop_size_max = 7
    min_gc = 25
    max_gc = 65
    key_size = 20
    key_num = 8
    
    constraints = Constraints(payload_size=payload_size, payload_num=payload_num, max_hom=max_hom, \
                              max_hairpin=max_hairpin, min_gc=min_gc, max_gc=max_gc, \
                              key_size=key_size, key_num=key_num, loop_size_min=loop_size_min, \
                              loop_size_max=loop_size_max)

    number_of_runs = 50

    hyp_tuning = HyperparameterTuning(number_of_runs)

    shape_values = {'hom': [10, 20, 30, 40, 50], \
                    'hairpin': [10, 20, 30, 40, 50], \
                    'similarity': [10, 20, 30, 40, 50], \
                    'gcContent': [10, 20, 30, 40, 50], \
                    'noKeyInPayload': [10, 20, 30, 40, 50]
                    }
    weight_values = {'hom': [1], \
                     'hairpin': [1], \
                     'similarity': [1], \
                     'gcContent': [1], \
                     'noKeyInPayload': [1],
                    } 
    with_constraints = {'hom', 'hairpin', 'gcContent', 'noKeyInPayload'}
    await hyp_tuning.grid_search(constraints, shape_values, weight_values, with_constraints)

def corIndexGc(line):
    if (" 'gcContent': 7}" in line):
        return 0
    if (" 'gcContent': 10}" in line):
        return 1
    if (" 'gcContent': 13}" in line):
        return 2
    if (" 'gcContent': 16}" in line):
        return 3
    if (" 'gcContent': 19}" in line):
        return 4
    
def corIndexHom(line):
    if ("apes: {'hom': 50, " in line):
        return 0
    if ("apes: {'hom': 55, " in line):
        return 1
    if ("apes: {'hom': 60, " in line):
        return 2
    if ("apes: {'hom': 65, " in line):
        return 3
    if ("apes: {'hom': 70, " in line):
        return 4
    
def corIndexHairpin(line):
    if (", 'hairpin': 2, " in line):
        return 0
    if (", 'hairpin': 4, " in line):
        return 1
    if (", 'hairpin': 6, " in line):
        return 2
    if (", 'hairpin': 8, " in line):
        return 3
    if (", 'hairpin': 10, " in line):
        return 4
    
def corIndexSim(line):
    if (", 'similarity': 50, " in line):
        return 0
    if (", 'similarity': 55, " in line):
        return 1
    if (", 'similarity': 60, " in line):
        return 2
    if (", 'similarity': 65, " in line):
        return 3
    if (", 'similarity': 70, " in line):
        return 4
    
def corIndexKey(line):
    if (", 'noKeyInPayload': 15, " in line):
        return 0
    if (", 'noKeyInPayload': 20, " in line):
        return 1
    if (", 'noKeyInPayload': 35, " in line):
        return 2
    if (", 'noKeyInPayload': 40, " in line):
        return 3
    if (", 'noKeyInPayload': 45, " in line):
        return 4
    

def dataExtract():
    data = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    with open(os.path.join(os.path.dirname(__file__), "hypTun2.txt"), "r") as file:
        while line := file.readline():
          data[corIndexHom(line)][corIndexGc(line)] += int(line.split('totalSuccesses: ')[1])
    import seaborn as sn
    # plotting the heatmap 
    hm = sn.heatmap(data = data, cmap = 'Blues', yticklabels=[50,55,60,65,70], xticklabels=[7,10,13,16,19]) 
    hm.set(ylabel="Homopolymer", xlabel="GC-Content")
    hm.invert_yaxis()
    
    # saving the plotted heatmap inside of the figures folder
    figure = hm.get_figure()    
    figure.savefig('hyperparameters/figures/round_two/hom_gc_heatmap.png', dpi=400)
    print(data)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    # Uncomment the below line to generate the heatmap for the hyperparameter tuning results
    # dataExtract()
    


    