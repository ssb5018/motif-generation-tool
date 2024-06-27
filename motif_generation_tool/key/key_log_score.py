import numpy as np
from dna_language_specification.language import nucleotides, converse
from constraints.hairpin import Hairpin


class KeyLogScore:
    def __init__(self, constraints, hyperparams):
        self.keys = set()
        self.keys_list = []
        self.key_size = constraints.key_size
        self.payload_size = constraints.payload_size
        self.motif_size = constraints.motif_size

        # Similarity
        self.used_bases = np.zeros(self.key_size)

        # Hairpin
        self.max_hairpin = constraints.max_hairpin
        self.hairpin = Hairpin(constraints, hyperparams)

        # Homopolymer
        self.max_hom = constraints.max_hom
        self.cur_hom = 0

        # GC-content
        self.min_gc = constraints.min_gc
        self.max_gc = constraints.max_gc
        self.cur_gc_count = 0

        # Key
        self.key_size = constraints.key_size
        self.min_gc_count = -1
        self.max_gc_count = -1
        self.cur_key = ''

        # Hyperparameters
        self.hom_hyperparams = hyperparams.hom
        self.hairpin_hyperparams = hyperparams.hairpin
        self.gc_content_hyperparams = hyperparams.gc_content
        self.similarity_hyperparams = hyperparams.similarity

    ### Get Log Scores ###

    def get_all_log_scores(self, key, base):
        log_score = self.get_homopolymer_log_score(key, base)
        log_score += self.get_hairpin_log_score(key, base)
        log_score += self.get_gc_log_score(key, base)
        log_score += self.get_similarity_log_score(key, base)
        return log_score

    def get_homopolymer_log_score(self, key, base):
        cur_key = key + base
        return self.homopolymer_log_score(cur_key)

    def get_hairpin_log_score(self, key, base):
        cur_key = key + base
        return self.hairpin_log_score(cur_key)

    def get_gc_log_score(self, key, base):
        cur_key = key + base
        return self.key_gc_content_log_score(cur_key)

    def get_similarity_log_score(self, key, base):
        cur_key = key + base
        return self.similarity_log_score(cur_key)
    
    ### Add Base to current Key ###
    
    async def add_base(self, new_base):
        self.cur_key += new_base
        await self.update_homopolymer_stats(new_base)
        await self.update_gc_count_stats(new_base)

    async def update_homopolymer_stats(self, new_base):
        if len(self.cur_key) <= 1:
            self.cur_hom = len(self.cur_key)
            return
        if self.cur_key[len(self.cur_key) - 2] != new_base:
            self.cur_hom = 0
        self.cur_hom += 1

    async def update_gc_count_stats(self, new_base):
        self.cur_gc_count += 1 if new_base in ['G', 'C'] else 0

    ### Start New Key ###

    def start_new_key(self):
        self.cur_key = ''
        self.cur_gc_count = 0
        self.cur_hom = 0

    ### Add Keys ###

    async def add_key(self, new_key):
        if len(new_key) != self.key_size:
            return False
        if new_key not in self.keys:
            self.keys.add(new_key)
            self.keys_list.append(new_key)
            await self.add_motif_gc_and_similarity_stats(new_key)
    
    async def add_keys(self, new_keys):
        for new_key in new_keys:
            await self.add_key(new_key)

    async def add_motif_gc_and_similarity_stats(self, new_key):
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        for i in range(self.key_size):
            b = new_key[i]
            if self.used_bases[i] // 10**unit[b] % 10 != 1:
                self.used_bases[i] += 10**unit[b]

        gc_count = sum([1 if b in ['G', 'C'] else 0 for b in new_key])
        self.min_gc_count = gc_count if self.min_gc_count == -1 else min(self.min_gc_count,\
                                                                         gc_count)
        self.max_gc_count = max(self.max_gc_count, gc_count)

    ##### Log Score Calculations #####

    ### Homopolymer Log Score ###

    def homopolymer_log_score(self, cur_key):
        max_hom_len = max(self.homopolymer_in_key(cur_key), \
                          self.homopolymer_within_motif_edge_cases(cur_key))
        if max_hom_len == 0:
            return 0
        if max_hom_len > self.max_hom:
            return -np.inf
        return - self.hom_hyperparams.shape**(max_hom_len / self.max_hom) + 1

    def homopolymer_in_key(self, cur_key):
        if len(cur_key) <= 1:
            return len(cur_key)
        return self.cur_hom + 1 if cur_key[len(cur_key) - 1] == cur_key[len(cur_key) - 2] else 1

    def get_first_and_last_key_pos_info(self, cur_key):
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        first_key_pos_used_bases = self.used_bases[0]
        last_key_pos_used_bases = self.used_bases[-1]
        cur_first_base = cur_key[0]
        if first_key_pos_used_bases // 10**unit[cur_first_base] % 10 != 1:
            first_key_pos_used_bases += 10**unit[cur_first_base]
        if len(cur_key) == self.key_size:
            cur_end_base = cur_key[self.key_size - 1]
            if last_key_pos_used_bases // 10**unit[cur_end_base] % 10 != 1:
                last_key_pos_used_bases += 10**unit[cur_end_base]
        same_unused_base = 0
        unused_bases_first_pos = 0
        unused_bases_last_pos = 0
        for b in nucleotides:
            first_key_pos_used_base = first_key_pos_used_bases // 10**unit[b] % 10 == 1
            last_key_pos_used_base = last_key_pos_used_bases // 10**unit[b] % 10 == 1
            if not (first_key_pos_used_base or last_key_pos_used_base):
                same_unused_base += 1
            if not first_key_pos_used_base:
                unused_bases_first_pos += 1
            if not last_key_pos_used_base:
                unused_bases_last_pos += 1
        return same_unused_base, unused_bases_first_pos, unused_bases_last_pos
    
    def homopolymer_within_motif_edge_cases(self, cur_key):
        if len(cur_key) == 0 or (len(cur_key) > 1 and len(cur_key) < self.key_size):
            return 0

        if self.max_hom != 1 and (self.max_hom != 2 or self.payload_size != 1): 
            return 0

        same_unused_base, unused_bases_first_pos, unused_bases_last_pos \
                                        = self.get_first_and_last_key_pos_info(cur_key)

        # max_hom = 1, make sure end of keys and start of keys not all nucleotides
        if self.max_hom == 1:
            if unused_bases_first_pos == 0 or unused_bases_last_pos == 0:
                return np.inf
            # payload_size = 1: make sure end and start of keys do not use at least one base
            if self.payload_size == 1 and same_unused_base == 0:
                return np.inf 
            # payload_size = 2: make sure all end and start bases of keys 
            # do not use at least 2 unique bases
            if self.payload_size == 2 and same_unused_base == 1 \
                                      and unused_bases_first_pos == 1 \
                                      and unused_bases_last_pos == 1:
                return np.inf
        # payload_size = 1, max_hom = 2, start and end should not use all bases
        elif self.max_hom == 2 and self.payload_size == 1:
            if unused_bases_first_pos == 4 and unused_bases_last_pos == 4:
                return np.inf
        return 0

    ### Key GC Content Log Score ###

    def key_gc_content_log_score(self, cur_key):
        key_gc_content = self.key_gc_content_within_motif(cur_key)
        return key_gc_content

    def key_gc_content_within_motif(self, cur_key):
        key_gc_count = self.cur_gc_count + 1 if cur_key[len(cur_key) - 1] in ['G', 'C'] \
                                         else self.cur_gc_count
        log_score = 0

        # with itself
        cur_motif_size = len(cur_key) * 2
        min_gc_count = np.ceil(self.min_gc * self.motif_size / 100)
        max_gc_count = np.floor(self.max_gc * self.motif_size / 100)
        if key_gc_count * 2 > max_gc_count or key_gc_count * 2 + self.motif_size - cur_motif_size\
                                                                                   < min_gc_count:
            return -np.inf

        weight = self.gc_content_hyperparams.shape**(cur_motif_size / self.motif_size) - 1
        min_gc_content = (100 * (key_gc_count * 2)) / cur_motif_size
        max_gc_content = (100 * (key_gc_count * 2)) / cur_motif_size
        log_score = max(log_score, weight * (self.min_gc - min_gc_content))
        log_score = max(log_score, weight * (max_gc_content - self.max_gc))

        if self.min_gc_count == -1 or self.max_gc_count == -1:
            return -log_score

        # with other keys
        cur_motif_size = len(cur_key) + self.key_size
        weight = self.gc_content_hyperparams.shape**(cur_motif_size / self.motif_size) - 1
        min_gc_content = (100 * (key_gc_count + self.min_gc_count)) / cur_motif_size
        max_gc_content = (100 * (key_gc_count + self.max_gc_count)) / cur_motif_size
        log_score = max(log_score, weight * (self.min_gc - min_gc_content))
        log_score = max(log_score, weight * (max_gc_content - self.max_gc))

        return -log_score

    ### Hairpin Log Score ###

    def hairpin_log_score(self, cur_key):
        log_score = self.hairpin.forward_hairpin_log_score(cur_key, self.keys_list, set(), \
                                                            is_key=True) + \
                    self.hairpin.backward_hairpin_log_score(cur_key, self.keys_list, set(), \
                                                            is_key=True)
        return log_score

    ### Similarity Log Score ###

    def similarity_log_score(self, cur_key):
        if not cur_key:
            return 0
        window_size = 0
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        for i in range(self.max_hairpin):
            index = len(cur_key) - 1 - i
            if self.used_bases[index] + 10**unit[cur_key[index]] >= 1111:
                window_size += 1
            else:
                break
        return - self.similarity_hyperparams.shape**(window_size / self.max_hairpin) + 1

