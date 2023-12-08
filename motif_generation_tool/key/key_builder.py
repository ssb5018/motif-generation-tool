from .key_log_score import KeyLogScore
from ..dna_language_specification.language import nucleotides

import numpy as np

class KeyBuilder:
    def __init__(self, constraints, hyperparameters):
        self.constraints = constraints
        
        self.hyperparams = hyperparameters

        self.key_num = constraints.key_num
        self.key_size = constraints.key_size

        self.key_log_score = KeyLogScore(constraints, hyperparameters)

    ### Build Keys ###

    async def build_key(self, with_constraints):
        key = ''

        self.key_log_score.start_new_key()
        for _ in range(self.key_size):
            
            log_scores = [0] * 4
            index_map = {'A':0, 'T':1, 'C':2, 'G':3}
            for n in nucleotides:
                index = index_map[n]
                log_scores[index] = 0
                for constraint in with_constraints:
                    if constraint == 'hom':
                        weight = self.hyperparams.hom.weight
                        hom_log_score = self.key_log_score.get_homopolymer_log_score(key, n)
                        log_scores[index] += weight * hom_log_score
                        if log_scores[index] == -np.inf:
                            break
                    elif constraint == 'gcContent':
                        weight = self.hyperparams.gc_content.weight
                        gc_log_score = self.key_log_score.get_gc_log_score(key, n)
                        log_scores[index] += weight * gc_log_score
                        if log_scores[index] == -np.inf:
                            break
                    elif constraint == 'hairpin':
                        weight = self.hyperparams.hairpin.weight
                        hairpin_log_score = self.key_log_score.get_hairpin_log_score(key, n)
                        log_scores[index] += weight * hairpin_log_score
                        if log_scores[index] == -np.inf:
                            break
                        weight = self.hyperparams.similarity.weight
                        similarity_log_score = self.key_log_score.get_similarity_log_score(key, n)
                        log_scores[index] += weight * similarity_log_score
            
            if not with_constraints:
                p = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                log_scores = np.array(log_scores)
                p = np.exp(log_scores)
                if not p.any():
                    return False
                p /= p.sum()

            new_nucleotide = np.random.choice(nucleotides, p=p)
        
            key += new_nucleotide
            await self.key_log_score.add_base(new_nucleotide)

        await self.key_log_score.add_key(key)
        return key
    
    async def build_all_keys(self, with_constraints):
        """This function attempts to build a key respecting the thresholds related to
        the constraints listed in `with_constraints` a total of `key_num` number of times
        
        Parameters
        ----------
        with_constraints: set of str
            Set of strings containing a selection of the following constraints: 
            'hairpin', 'hom', 'gcContent'. Those will be the constraints that the
            palyoads will have to conform to.
        
        Returns
        ----------
        all_keys_list: list of str
            List of keys conforming to the constraints `with_constraints`.
        """

        all_keys = set()
        all_keys_list = []
        for _ in range(self.key_num):
            key = await self.build_key(with_constraints)
            if not key:
                continue
            if not key in all_keys:
                all_keys_list.append(key)
            all_keys.add(key)
        return all_keys_list


async def main():
    from ..constraints.constraints import Constraints
    from ..hyperparameters.hyperparameters import Hyperparameters
    payload_size = 8
    payload_num = 5
    max_hom = 1
    max_hairpin = 1
    loop_size_min = 6
    loop_size_max = 7
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

        keyBuilder = KeyBuilder(constraints, hyperparams)
        keys = await keyBuilder.build_all_keys(with_constraints)

        if not keys:
            continue
        num_successful_keys += 1
        print(keys)

    print('Number of key lists conforming to the constraints is ', num_successful_keys)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
