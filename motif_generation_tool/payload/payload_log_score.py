import numpy as np
from dna_language_specification.language import nucleotides, converse
from constraints.hairpin import Hairpin


class PayloadLogScore:
    def __init__(self, constraints, hyperparams):
        
        self.payloads = set()
        self.motif_size = constraints.motif_size

        # Hairpin
        self.max_hairpin = constraints.max_hairpin
        self.hairpin = Hairpin(constraints, hyperparams)

        # Homopolymer
        self.max_hom = constraints.max_hom

        # GC-content
        self.min_gc = constraints.min_gc
        self.max_gc = constraints.max_gc

        # Keys
        self.keys = []
        self.key_size = constraints.key_size
        self.start_key_hom = []
        self.end_key_hom = []
        self.max_start_key_hom = {'A':0, 'T':0, 'C':0, 'G':0}
        self.max_end_key_hom = {'A':0, 'T':0, 'C':0, 'G':0}
        self.whole_key_hom_indices = {'A':-1, 'T':-1, 'C':-1, 'G':-1}
        self.min_key_gc_count = -1
        self.max_key_gc_count = -1

        # Payload
        self.payload_size = constraints.payload_size
        self.max_start_payload_hom = {'A':0, 'T':0, 'C':0, 'G':0}
        self.max_end_payload_hom = {'A':0, 'T':0, 'C':0, 'G':0}

        self.used_bases = np.zeros(self.payload_size)

        # Hyperparameters
        self.hom_hyperparams = hyperparams.hom
        self.hairpin_hyperparams = hyperparams.hairpin
        self.gc_content_hyperparams = hyperparams.gc_content
        self.similarity_hyperparams = hyperparams.similarity
        self.no_key_in_payload_hyperparams = hyperparams.no_key_in_payload

    ### Get Log Score ###

    def get_all_log_score(self, payload, base):
        log_score = self.get_homopolymer_log_score(payload, base)
        log_score += self.get_hairpin_log_score(payload, base)
        log_score += self.get_gc_log_score(payload, base)
        log_score += self.get_similarity_log_score(payload, base)
        log_score += self.get_no_key_in_payload_log_score(payload, base)
        return log_score

    def get_homopolymer_log_score(self, payload, base):
        cur_payload = payload + base
        return self.homopolymer_log_score(cur_payload)

    def get_hairpin_log_score(self, payload, base):
        cur_payload = payload + base
        return self.hairpin_log_score(cur_payload)

    def get_gc_log_score(self, payload, base):
        cur_payload = payload + base
        return self.motif_gc_content_log_score(cur_payload)

    def get_similarity_log_score(self, payload, base):
        cur_payload = payload + base
        return self.similarity_log_score(cur_payload)
    
    def get_no_key_in_payload_log_score(self, payload, base):
        cur_payload = payload + base
        return self.no_key_in_payload_log_score(cur_payload)

    ##### Add Payloads and Keys #####

    ### Add Payloads ###
    
    async def add_payload(self, new_payload):
        if len(new_payload) != self.payload_size:
            return False
        self.payloads.add(new_payload)
        await self.add_hom_and_similarity_stats(new_payload)
    
    async def add_payloads(self, new_payloads):
        for new_payload in new_payloads:
            await self.add_payload(new_payload)

    ### Add Keys ###

    async def add_keys(self, keys):
        self.keys = keys if isinstance(keys, list) else list(keys)
        await self.generate_key_pre_stats()
    
    ##### Pre-Stats #####

    ### Key Pre-Stats ###

    async def generate_key_pre_stats(self):
        for j in range(len(self.keys)):
            key = self.keys[j]
            cur_base = key[0]
            cur_hom = 1
            cur_gc_count = 1 if cur_base in ['G', 'C'] else 0
            is_start = True

            for i in range(1, self.key_size):
                b = key[i]

                # Update start homopolymer
                if cur_base != b:
                    if is_start:
                        is_start = False
                        self.max_start_key_hom[cur_base] = max(self.max_start_key_hom[cur_base],\
                                                               cur_hom)
                        self.start_key_hom.append(cur_hom)
                    cur_base = b
                    cur_hom = 0
                
                cur_gc_count += 1 if b in ['G', 'C'] else 0

                cur_hom += 1

            # Update homopolymer
            if is_start:
                self.whole_key_hom_indices[cur_base] = j
                self.start_key_hom.append(cur_hom)
                self.max_start_key_hom[cur_base] = max(self.max_start_key_hom[cur_base], cur_hom)
            self.max_end_key_hom[cur_base] = max(self.max_end_key_hom[cur_base], cur_hom)
            self.end_key_hom.append(cur_hom)
        
            # Update overall GC-count
            self.min_key_gc_count = cur_gc_count if self.min_key_gc_count == -1 \
                                                 else min(self.min_key_gc_count, cur_gc_count)
            self.max_key_gc_count = max(self.max_key_gc_count, cur_gc_count)

    ### Payload Pre-Stats ###

    async def add_hom_and_similarity_stats(self, payload):
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        is_start = True
        cur_payload = payload[0]
        cur_hom = 1
        for i in range(1, self.payload_size):
            b = payload[i]

            # Update similarity stats
            if self.used_bases[i] // 10**unit[b] % 10 != 1:
                self.used_bases[i] += 10**unit[b]

            # Update homopolymer stats
            if cur_payload != b:
                if is_start:
                    is_start = False
                    self.max_start_payload_hom[cur_payload] = \
                                    max(self.max_start_payload_hom[cur_payload], cur_hom)
                cur_hom = 0
                cur_payload = b
            cur_hom += 1

        # Update homopolymer stats
        if is_start:
            self.max_start_payload_hom[cur_payload] = max(self.max_start_payload_hom[cur_payload],\
                                                          cur_hom)
        self.max_end_payload_hom[cur_payload] = max(self.max_end_payload_hom[cur_payload], cur_hom)

    ##### Log Score Calculations #####

    ### Homopolymer Log Score ###

    def calculate_hom_log_score(self, hom_length):
        if hom_length > self.max_hom:
            return -np.inf
        return - self.hom_hyperparams.shape**(hom_length/self.max_hom) + 1

    def homopolymer_log_score(self, cur_payload):
        max_hom_length = self.max_homopolymer_length(cur_payload)
        return self.calculate_hom_log_score(max_hom_length)
    
    def get_start_hom_count(self, payload, base):
        cur_start = base
        cur_start_hom = 0
        for i in range(len(payload)):
            b = payload[i]
            if cur_start != b:
                break
            cur_start_hom += 1
        return cur_start_hom

    def max_homopolymer_length(self, cur_payload):
        cur_hom = 1
        cur_base = cur_payload[-1]
        for i in range(len(cur_payload) - 2, -1, -1):
            b = cur_payload[i]
            if cur_base != b:
                break
            cur_hom += 1
        added_base = cur_payload[-1]

        if not self.start_key_hom:
            return cur_hom

        # Start and end keys
        if cur_hom == self.payload_size:
            if self.whole_key_hom_indices[added_base] != -1:
                # Whole motif homomopolymer with only one key
                if len(self.keys) == 1:
                    return np.inf
                whole_key_index = self.whole_key_hom_indices[added_base]
                prev_key_index = (whole_key_index - 1) % len(self.keys)
                next_key_index = (whole_key_index + 1) % len(self.keys)
                end_prev_key_hom = self.end_key_hom[prev_key_index] \
                                   if self.keys[prev_key_index][self.key_size - 1] == added_base \
                                   else 0
                start_next_key_hom = self.start_key_hom[next_key_index] \
                                    if self.keys[next_key_index][0] == added_base \
                                    else 0
                hom_len = end_prev_key_hom + 2 * self.key_size + start_next_key_hom + \
                          self.payload_size * 3
                return hom_len
            else:
                hom_len = self.max_end_key_hom[added_base] + self.payload_size + \
                          self.max_start_key_hom[added_base]
                return hom_len
        # Start keys
        elif cur_hom == len(cur_payload):
            if self.whole_key_hom_indices[added_base] != -1:
                if self.max_end_payload_hom[added_base] == self.payload_size:
                    whole_key_index = self.whole_key_hom_indices[added_base]
                    prev_key_index = (whole_key_index - 1) % len(self.keys)
                    end_prev_key_hom = self.end_key_hom[prev_key_index] \
                                    if self.keys[prev_key_index][self.key_size - 1] == added_base \
                                    else 0
                    hom_len = cur_hom + 2 * self.key_size + 2 * self.payload_size + \
                              end_prev_key_hom 
                    return hom_len
                else:
                    hom_len = cur_hom + self.key_size + self.max_end_payload_hom[added_base]
                    return hom_len
            else:
                hom_len = cur_hom + self.max_end_key_hom[added_base]
                return hom_len
        # End keys
        elif len(cur_payload) == self.payload_size:
            if self.whole_key_hom_indices[added_base] != -1:
                if self.max_start_payload_hom[added_base] == self.payload_size:
                    whole_key_index = self.whole_key_hom_indices[added_base]
                    next_key_index = (whole_key_index + 1) % len(self.keys)
                    start_next_key_hom = self.start_key_hom[next_key_index] \
                                         if self.keys[next_key_index][0] == added_base else 0
                    hom_len = cur_hom + 2 * self.key_size + 2 * self.payload_size + \
                              start_next_key_hom
                    return hom_len
                else:
                    cur_start_hom = self.get_start_hom_count(cur_payload, added_base)
                    max_start_payload_hom = max(self.max_start_payload_hom[added_base], \
                                                cur_start_hom)
                    hom_len = cur_hom + self.key_size + max_start_payload_hom
                    return hom_len
            else:
                hom_len = cur_hom + self.max_start_key_hom[added_base]
                return hom_len
        return cur_hom

    ### Motif GC Content Log Score ###

    def motif_gc_content_log_score(self, cur_payload):
        gc_count = sum([1 if b in ['G', 'C'] else 0 for b in cur_payload])

        if len(self.keys) == 0:
            weight = - self.gc_content_hyperparams.shape**(len(cur_payload) / self.motif_size) + 1
            return weight * 100 * gc_count / len(cur_payload)
        
        cur_motif_size = len(cur_payload) + self.key_size * 2
        weight = self.gc_content_hyperparams.shape**(cur_motif_size / self.motif_size) - 1
        min_gc_content = (100 * (gc_count + self.min_key_gc_count * 2)) / cur_motif_size
        max_gc_content = (100 * (gc_count + self.max_key_gc_count * 2)) / cur_motif_size
        log_score = 0
        log_score = max(log_score, weight * (self.min_gc - min_gc_content))
        log_score = max(log_score, weight * (max_gc_content - self.max_gc))
        if cur_motif_size == self.motif_size and log_score != 0:
            return -np.inf
        return -log_score

    ### Hairpin Log Score ###

    def hairpin_log_score(self, cur_payload):
        log_score = self.hairpin.forward_hairpin_log_score(cur_payload, self.payloads, self.keys) \
                   + self.hairpin.backward_hairpin_log_score(cur_payload, self.payloads, self.keys)
        return log_score

    ### Similarity Log Score ###

    def similarity_log_score(self, cur_payload):
        if not cur_payload:
            return 0
        window_size = 0
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        for i in range(self.max_hairpin):
            index = len(cur_payload) - 1 - i
            if self.used_bases[index] + 10**unit[cur_payload[index]] >= 1111:
                window_size += 1
            else:
                break
        return - self.similarity_hyperparams.shape**(window_size / self.max_hairpin) + 1
    
    ### No Key in Payload Log Score ###

    def no_key_in_payload_log_score(self, cur_payload):
        if not cur_payload:
            return 0
        window_size = 0
        unit = {'A':0, 'T':1, 'C':2, 'G':3}
        for key in self.keys:
            for i in range(1, min(len(cur_payload), self.key_size)):
                index = len(cur_payload) - 1 - i
                if cur_payload[index:] in key:
                    window_size = max(window_size, i)
                    if window_size == self.key_size:
                        return -np.inf
                else:
                    break
        return - self.no_key_in_payload_hyperparams.shape**(window_size / self.key_size) + 1

