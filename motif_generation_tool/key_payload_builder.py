from .payload.payload_builder import PayloadBuilder
from .key.key_builder import KeyBuilder

import numpy as np

class KeyPayloadBuilder:
    def __init__(self, constraints, hyperparameters):
        self.constraints = constraints
        self.hyperparameters = hyperparameters

    async def build_keys_and_payloads(self, with_constraints):
        """This function attempts to build keys and payloads respecting the thresholds related to
        the constraints listed in `with_constraints` a total of `key_num` and 
        `payload_num` number of times respectively.
        
        Parameters
        ----------
        with_constraints: set of str
            Set of strings containing a selection of the following constraints: 
            'hairpin', 'hom', 'gcContent'. Those will be the constraints that the
            palyoads will have to conform to.
        
        Returns
        ----------
        keys: list of str or bool
            List of keys conforming to the constraints `with_constraints`. If no such
            key could be generated, it is False.
        payload: set of str or bool
            Set of payloads conforming to the constraints `with_constraints`. If no such
            key could be generated, it is False.
        """

        key_builder = KeyBuilder(self.constraints, self.hyperparameters)
        keys = await key_builder.build_all_keys(with_constraints)
        if not keys:
            return False, False
        payload_builder = PayloadBuilder(self.constraints, self.hyperparameters)
        await payload_builder.add_keys(keys)
        payloads = await payload_builder.build_all_payloads(with_constraints)
        if not payloads:
            return False, False
        return keys, payloads

    def get_motifs(self, keys, payloads):
        """This function returns the set of motifs corresponding to the keys and payloads inputted.
        
        Parameters
        ----------
        keys: list of str
            List of keys.
        payload: set of str
            Set of payloads.
        
        Returns
        ----------
        motifs: set of str or bool
            Set of all motifs built using the provided keys and payloads.
        """
        motifs = set()
        for payload in payloads:
            for i in range(len(keys)):
                motif1 = keys[i] + payload + keys[i]
                motif2 = keys[i] + payload + keys[(i + 1) % len(keys)]
                motifs.add(motif1)
                motifs.add(motif2)
        return motifs
    

async def main():
    from .constraints.constraints import Constraints
    from .hyperparameters.hyperparameters import Hyperparameters
    payload_size = 8
    payload_num = 5
    max_hom = 1
    max_hairpin = 1
    loop_size_min = 1
    loop_size_max = 2
    min_gc = 25
    max_gc = 60
    key_size = 1
    key_num = 5
    
    constraints = Constraints(payload_size=payload_size, payload_num=payload_num, \
                              max_hom=max_hom, max_hairpin=max_hairpin, \
                              min_gc=min_gc, max_gc=max_gc, key_size=key_size, \
                              key_num=key_num, loop_size_min=loop_size_min, \
                              loop_size_max=loop_size_max)

    num_rounds = 1
    num_successful_keys = 0
    with_constraints = {'hom', 'gcContent', 'hairpin'}
    for i in range(num_rounds):
        shapes = {'hom': 70, 'gcContent': 20, 'hairpin': 8, 'similarity': 50}
        weights = {'hom': 1, 'gcContent': 1, 'hairpin': 1, 'similarity': 1}
        hyperparams = Hyperparameters(shapes, weights)

        keyPayloadBuilder = KeyPayloadBuilder(constraints, hyperparams)
        keys, payloads = await keyPayloadBuilder.build_keys_and_payloads(with_constraints)
        if not (keys and payloads):
            continue
        print('keys: ', keys)
        print('payloads: ', payloads)
        num_successful_keys += 1

    print('Number of motif sets conforming to the constraints is ', num_successful_keys)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
