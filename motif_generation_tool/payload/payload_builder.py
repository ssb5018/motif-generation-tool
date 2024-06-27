from .payload_log_score import PayloadLogScore
from constraints.constraints import Constraints
from hyperparameters.hyperparameters import Hyperparameters
from dna_language_specification.language import nucleotides

import numpy as np


class PayloadBuilder:
    def __init__(self, constraints, hyperparameters):
        self.constraints = constraints

        self.hyperparams = hyperparameters

        self.payload_num = constraints.payload_num
        self.payload_size = constraints.payload_size

        self.payload_log_score = PayloadLogScore(constraints, hyperparameters)
        
    async def add_keys(self, keys):
        await self.payload_log_score.add_keys(keys)

    ### Build Payload ###

    async def build_payload(self, with_constraints):
        payload = ''

        for _ in range(self.payload_size):
            
            log_scores = [0] * 4
            index_map = {'A':0, 'T':1, 'C':2, 'G':3}
            for n in nucleotides:
                index = index_map[n]
                log_scores[index] = 0
                for constraint in with_constraints:
                    if constraint == 'hom':
                        weight = self.hyperparams.hom.weight
                        hom_log_score = self.payload_log_score.get_homopolymer_log_score(payload, n)
                        log_scores[index] += weight * hom_log_score
                        if log_scores[index] == -np.inf:
                            break
                    elif constraint == 'gcContent':
                        weight = self.hyperparams.gc_content.weight
                        gc_log_score = self.payload_log_score.get_gc_log_score(payload, n)
                        log_scores[index] += weight * gc_log_score
                        if log_scores[index] == -np.inf:
                            break
                    elif constraint == 'hairpin':
                        weight = self.hyperparams.hairpin.weight
                        hairpin_log_score = self.payload_log_score.get_hairpin_log_score(payload, n)
                        log_scores[index] += weight * hairpin_log_score
                        if log_scores[index] == -np.inf:
                            break
                        weight = self.hyperparams.similarity.weight
                        similarity_log_score = \
                                        self.payload_log_score.get_similarity_log_score(payload, n)
                        log_scores[index] += weight * similarity_log_score
                    elif constraint == 'noKeyInPayload':
                        weight = self.hyperparams.no_key_in_payload.weight
                        no_key_in_payload_log_score = self.payload_log_score.get_no_key_in_payload_log_score(payload, n)
                        log_scores[index] += weight * no_key_in_payload_log_score
                        if log_scores[index] == -np.inf:
                            break
            
            if not with_constraints: 
                p = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                log_scores = np.array(log_scores)
                p = np.exp(log_scores)
                if not p.any():
                    await self.payload_log_score.add_payload(payload)
                    return False
                p /= p.sum()
            
            new_nucleotide = np.random.choice(nucleotides, p=p)

            payload += new_nucleotide

        await self.payload_log_score.add_payload(payload)

        return payload
    
    async def build_all_payloads(self, with_constraints):
        """This function attempts to build a payload respecting the thresholds related to
        the constraints listed in `with_constraints` a total of `payload_num` number of times
        
        Parameters
        ----------
        with_constraints: set of str
            Set of strings containing a selection of the following constraints: 
            'hairpin', 'hom, 'gcContent'. Those will be the constraints that the
            palyoads will have to conform to.
        
        Returns
        ----------
        all_payloads: set of str
            Set of payloads conforming to the constraints `with_constraints`.
        """
        all_payloads = set()
        for _ in range(self.payload_num):
            payload = await self.build_payload(with_constraints)
            if not payload:
                continue
            all_payloads.add(payload)
        return all_payloads


async def main():
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

    num_rounds = 10
    num_successful_keys = 0
    with_constraints = {'hom', 'gcContent', 'hairpin'}
    for i in range(num_rounds):
        shapes = {'hom': 5, 'gcContent': 5, 'hairpin': 5, 'similarity': 5}
        weights = {'hom': 1, 'gcContent': 1, 'hairpin': 1, 'similarity': 1}
        hyperparams = Hyperparameters(shapes, weights)

        keys = ['A']
        payload_builder = PayloadBuilder(constraints, hyperparams)
        await payload_builder.add_keys(keys)
        payloads = await payload_builder.build_all_payloads(with_constraints)

        if not payloads:
            continue
        print(payloads)
        num_successful_keys += 1

    print('Number of payload sets conforming to the constraints is ', num_successful_keys)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
