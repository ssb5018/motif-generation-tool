import numpy as np
from ..dna_language_specification.language import nucleotides, converse
from ..hyperparameters.hyperparameters import Hyperparameters

class Hairpin:
    def __init__(self, constraints, hyperparams=None):
        self.window_size = constraints.motif_size - constraints.key_size

        # Hairpin
        self.max_hairpin = constraints.max_hairpin
        self.stem_out_bounds_len = constraints.max_hairpin + 1
        self.loop_size_min = constraints.loop_size_min
        self.loop_size_max = constraints.loop_size_max

        # Key
        self.key_size = constraints.key_size

        # Payload
        self.payload_size = constraints.payload_size

        # Hyperparameters
        if hyperparams:
            self.hairpin_hyperparams = hyperparams.hairpin
        else:
            self.hairpin_hyperparams = Hyperparameters().hairpin

    ##### Helper Functions #####

    def calculate_hairpin_log_score(self, stem_length):
        return - self.hairpin_hyperparams.shape**(stem_length / (self.max_hairpin)) + 1

    def is_in_cur_elem(self, info, pos):
        return pos >= 0 and pos < info['elem1Size']

    def is_in_elem2(self, info, pos):
        return (pos % self.window_size) >= info['elem1Size']

    def is_in_other_elem1(self, info, pos):
        return not (self.is_in_cur_elem(info, pos) or self.is_in_elem2(info, pos))

    def is_in_same_elem1(self, info, pos1, pos2):
        if self.is_in_elem2(info, pos1) or self.is_in_elem2(info, pos2):
            return False
        if (pos1 < 0 and pos2 >= 0) or (pos2 < 0 and pos1 >= 0):
            return False
        if pos1 < 0:
            pos1 -= 1
        if pos2 < 0:
            pos2 -= 1
        return int(pos1 / self.window_size) == int(pos2 / self.window_size)

    def is_in_same_elem2(self, info, pos1, pos2):
        if not (self.is_in_elem2(info, pos1) and self.is_in_elem2(info, pos2)):
            return False
        if (pos1 < 0 and pos2 >= 0) or (pos2 < 0 and pos1 >= 0):
            return False
        if pos1 < 0:
            pos1 -= 1
        if pos2 < 0:
            pos2 -= 1
        return int(pos1 / self.window_size) == int(pos2 / self.window_size)

    def get_key_at_pos(self, info, pos):
        if not info['isKey']:
            pos = pos + self.key_size
            keyElems = info['elems2']
            totalKeyNum = len(keyElems) * 2
        else:
            keyElems = info['elems1']
            totalKeyNum = len(keyElems) * 2 + 2
        if totalKeyNum == 0:
            return ""
        if pos < 0:
                pos = pos - self.key_size
        keyNum = (info['firstKey'] + int(pos / self.window_size)) % totalKeyNum
        if keyNum >= len(keyElems) * 2:
            return info['curElem']
        return keyElems[int(keyNum / 2)]
    
    def stem_contains_only_one_key(self, info, start_pos):
        if self.stem_contains_no_key(info, start_pos):
            return False
        if self.stem_out_bounds_len <= self.payload_size + 1:
            return True
        if self.stem_out_bounds_len > 2 * self.payload_size + self.key_size:
            return False
        if info['isKey']:
            if self.is_in_elem2(info, start_pos):
                return self.stem_out_bounds_len + ((start_pos % self.window_size) - self.key_size)\
                        <= 2 * self.payload_size + self.key_size
            return self.stem_out_bounds_len + (start_pos % self.window_size) <= self.window_size
        if not self.is_in_elem2(info, start_pos):
            return self.stem_out_bounds_len + (start_pos % self.window_size) <= \
                    2 * self.payload_size + self.key_size
        return self.stem_out_bounds_len + ((start_pos % self.window_size) - self.payload_size) \
                <= self.window_size

    def stem_contains_no_key(self, info, start_pos):
        end_pos = start_pos + self.stem_out_bounds_len - 1
        if info['isKey']:
            return self.is_in_same_elem2(info, start_pos, end_pos)
        return self.is_in_same_elem1(info, start_pos, end_pos)
    
    def stems_in_same_key_even_after_shift(self, info, start_pos):
        if self.stem_contains_no_key(info, start_pos):
            return True
        if not self.stem_contains_only_one_key(info, start_pos):
            return False
        if info['isKey']:
            if self.is_in_elem2(info, start_pos):
                keyPos = start_pos + self.payload_size - ((start_pos % self.window_size) - \
                                                          self.key_size)
            else: 
                keyPos = start_pos
            return self.get_key_at_pos(info, keyPos) == self.get_key_at_pos(info, keyPos + \
                                                                            self.window_size)
        if not self.is_in_elem2(info, start_pos):
            keyPos = start_pos + self.payload_size - (start_pos % self.window_size)
        else:
            keyPos = start_pos
        return self.get_key_at_pos(info, keyPos) == self.get_key_at_pos(info, keyPos + \
                                                                        self.window_size)
    
    ##### Send Current Stem 1 and Stem 2 Positions to Corresponding Payloads and Keys #####

    def send_to_all_check(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins, \
                          curE1_1="", curE1_2="", curE2_1="", curE2_2=""):
        if hairpins == -np.inf:
            return True, hairpins
        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ

        if curJ < 0 and stem_length > 0:
            if stem_length >= self.stem_out_bounds_len:
                hairpins = -np.inf
                return True, hairpins
            hairpins.append(self.calculate_hairpin_log_score(stem_length))
            return False, hairpins

        first_keys = [info['firstKey']]
        if info['firstKey'] == -1:
            if info['isKey']:
                # Only 1 key
                if not info['elems1']:
                    first_keys = [1]
                else:
                    first_keys = [len(info['elems1']) * 2, len(info['elems1']) * 2 + 1]
            elif info['elems2']:
                if len(info['elems2']) == 1:
                    first_keys = [len(info['elems2']) * 2 + 1]
                else:
                    first_keys = [i for i in range(len(info['elems2'] * 2))]
        for i in range(len(first_keys)):
            info['firstKey'] = first_keys[i]
            for j in range(curJ, -1, -1):
                was_sent_to_all_pos1_e1, hairpins = self.send_to_all_pos1_elem1(info, curJ,\
                                                                    stem1_start, stem2_start,\
                                                                    stem_length, hairpins, \
                                                                    curE1_1, curE1_2, \
                                                                    curE2_2)
                was_sent_to_all_pos1_e2, hairpins = self.send_to_all_pos1_elem2(info, curJ,\
                                                                            stem1_start, \
                                                                            stem2_start, \
                                                                            stem_length, \
                                                                            hairpins, curE2_1, \
                                                                            curE2_2, curE1_2)
                if hairpins == -np.inf or ((was_sent_to_all_pos1_e1 or was_sent_to_all_pos1_e2) \
                                                                and i == len(first_keys) - 1):
                    return True, hairpins
                elif (was_sent_to_all_pos1_e1 or was_sent_to_all_pos1_e2):
                    break
                curJ -= 1
        return False, hairpins

    def send_to_all_pos1_elem1(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins,\
                               curE1_1="", curE1_2="", curE2=""):
        if hairpins == -np.inf:
            return True, hairpins
        was_sent_to_e1_e1, hairpins = self.send_to_elem1_elem1(info, curJ, stem1_start, \
                                                                 stem2_start, stem_length, \
                                                                 hairpins, curE1_1, curE1_2)
        was_sent_to_e1_e2, hairpins = self.send_to_elem1_elem2(info, curJ, stem1_start, \
                                                                 stem2_start, stem_length, \
                                                                 hairpins, curE1_1, curE2)
        return was_sent_to_e1_e1 or was_sent_to_e1_e2, hairpins
    
    def send_to_key_key(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins):
        if hairpins == -np.inf:
            return True, hairpins
        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ

        if info['isKey'] and (self.is_in_elem2(info, stem1_pos) or \
                                self.is_in_elem2(info, stem2_pos)):
            return False, hairpins

        if not info['isKey'] and not (self.is_in_elem2(info, stem1_pos) and \
                                            self.is_in_elem2(info, stem2_pos)):
            return False, hairpins

        if info['isKey']:
            e1_1 = self.get_key_at_pos(info, stem1_pos)
            e1_2 = self.get_key_at_pos(info, stem2_pos)
            hairpins = self.hairpin_count_elem1_elem1(info, e1_1, e1_2, curJ, \
                                                        stem1_start, stem2_start, \
                                                        stem_length, hairpins)
        else:
            e2_1 = self.get_key_at_pos(info, stem1_pos)
            e2_2 = self.get_key_at_pos(info, stem2_pos)
            hairpins = self.hairpin_count_elem2_elem2(info, e2_1, e2_2, curJ, \
                                                        stem1_start, stem2_start, \
                                                        stem_length, hairpins)
        return True, hairpins
        

    def send_to_elem1_elem1(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins, \
                            curE1_1="", curE1_2=""):
        if hairpins == -np.inf:
            return True, hairpins
        if info['isKey']:
            return self.send_to_key_key(info, curJ, stem1_start, stem2_start, stem_length, \
                                        hairpins)

        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ

        if self.is_in_elem2(info, stem1_pos) or self.is_in_elem2(info, stem2_pos):
            return False, hairpins

        if curE1_1 and curE1_2:
            hairpins = self.hairpin_count_elem1_elem1(info, curE1_1, curE1_2, curJ, stem1_start, \
                                                      stem2_start, stem_length, hairpins)
        elif curE1_1:            
            if not self.is_in_cur_elem(info, stem2_pos):
                for e1 in info['elems1']:
                    if self.is_in_same_elem1(info, stem1_pos, stem2_pos) and curE1_1 != e1:
                        continue
                    hairpins = self.hairpin_count_elem1_elem1(info, curE1_1, e1, curJ, \
                                                            stem1_start, stem2_start, \
                                                            stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            hairpins = self.hairpin_count_elem1_elem1(info, curE1_1, info['curElem'], curJ, \
                                                    stem1_start, stem2_start, stem_length, \
                                                    hairpins)
        elif curE1_2:
            if not self.is_in_cur_elem(info, stem1_pos):
                for e1 in info['elems1']:
                    if self.is_in_same_elem1(info, stem1_pos, stem2_pos) and e1 != curE1_2:
                        continue
                    hairpins = self.hairpin_count_elem1_elem1(info, e1, curE1_2, curJ, \
                                                            stem1_start, stem2_start, \
                                                            stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            hairpins = self.hairpin_count_elem1_elem1(info, info['curElem'], curE1_2, curJ, \
                                                    stem1_start, stem2_start, stem_length, \
                                                    hairpins)
        else:
            pos1_pos2_in_curElem = self.is_in_cur_elem(info, stem1_pos) and \
                            self.is_in_cur_elem(info, stem2_pos)
            if (not pos1_pos2_in_curElem) and self.is_in_cur_elem(info, stem1_pos):
                for e1_2 in info['elems1']:
                    if self.is_in_same_elem1(info, stem1_pos, stem2_pos) and \
                                                                    info['curElem'] != e1_2:
                        continue
                    hairpins = self.hairpin_count_elem1_elem1(info, info['curElem'], e1_2, curJ, \
                                                            stem1_start, stem2_start, \
                                                            stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            elif (not pos1_pos2_in_curElem) and self.is_in_cur_elem(info, stem2_pos):
                for e1_1 in info['elems1']:
                    if self.is_in_same_elem1(info, stem1_pos, stem2_pos) and \
                                                                info['curElem'] != e1_1:
                        continue
                    hairpins = self.hairpin_count_elem1_elem1(info, e1_1, info['curElem'], curJ, \
                                                            stem1_start, stem2_start, \
                                                            stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            elif not (self.is_in_cur_elem(info, stem1_pos) or self.is_in_cur_elem(info, stem2_pos)):
                for e1_1 in info['elems1']:
                    for e1_2 in info['elems1']:
                        if self.is_in_same_elem1(info, stem1_pos, stem2_pos) and e1_1 != e1_2:
                            continue
                        hairpins = self.hairpin_count_elem1_elem1(info, e1_1, e1_2, curJ, \
                                                                stem1_start, stem2_start, \
                                                                stem_length, hairpins)
                        if hairpins == -np.inf:
                            return True, hairpins
            hairpins = self.hairpin_count_elem1_elem1(info, info['curElem'], info['curElem'], \
                                                    curJ, stem1_start, stem2_start, \
                                                    stem_length, hairpins)
        return True, hairpins

    def send_to_elem1_elem2(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins, \
                            curE1="", curE2=""):
        if hairpins == -np.inf:
            return True, hairpins
        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ

        if not ((not self.is_in_elem2(info, stem1_pos)) and self.is_in_elem2(info, stem2_pos)):
            return False, hairpins
        
        if info['isKey']:
            curE1 = self.get_key_at_pos(info, stem1_start)
        else:
            curE2 = self.get_key_at_pos(info, stem2_start)

        if curE1 and curE2:
            hairpins = self.hairpin_count_elem1_elem2(info, curE1, curE2, curJ, stem1_start, \
                                                      stem2_start, stem_length, hairpins)
        elif curE1:
            for e2 in info['elems2']:
                hairpins = self.hairpin_count_elem1_elem2(info, curE1, e2, curJ, stem1_start, \
                                                        stem2_start, stem_length, hairpins)
                if hairpins == -np.inf:
                    return True, hairpins
        elif curE2:
            if not self.is_in_cur_elem(info, stem1_pos):
                for e1 in info['elems1']:
                    hairpins = self.hairpin_count_elem1_elem2(info, e1, curE2, curJ, stem1_start,\
                                                            stem2_start, stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            hairpins = self.hairpin_count_elem1_elem2(info, info['curElem'], curE2, curJ, \
                                                    stem1_start, stem2_start, stem_length, \
                                                    hairpins)
        if not (curE2 or info['elems2']):
            _, hairpins = self.send_to_all_check(info, curJ-1, stem1_start, stem2_start, \
                                              stem_length, hairpins, curE1_1=curE1, curE2_2=curE2)
        
        return True, hairpins

    def send_to_all_pos1_elem2(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins,\
                               curE2_1="", curE2_2="", curE1=""):
        if hairpins == -np.inf:
            return True, hairpins
        was_sent_to_e2_e1, hairpins = self.send_to_elem2_elem1(info, curJ, stem1_start, \
                                                                 stem2_start, stem_length, \
                                                                 hairpins, curE2_1, curE1)
        was_sent_to_e2_e2, hairpins = self.send_to_elem2_elem2(info, curJ, stem1_start, \
                                                                 stem2_start, stem_length, \
                                                                 hairpins, curE2_1, curE2_2)
        return was_sent_to_e2_e1 or was_sent_to_e2_e2, hairpins

    def send_to_elem2_elem1(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins, \
                            curE2="", curE1=""):
        if hairpins == -np.inf:
            return True, hairpins
        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ
        if not (self.is_in_elem2(info, stem1_pos) and (not self.is_in_elem2(info, stem2_pos))):
            return False, hairpins
        
        if info['isKey']:
            curE1 = self.get_key_at_pos(info, stem2_start)
        else:
            curE2 = self.get_key_at_pos(info, stem1_start)
        if curE2 and curE1:
            hairpins = self.hairpin_count_elem2_elem1(info, curE2, curE1, curJ, stem1_start, \
                                                      stem2_start, stem_length, hairpins)
        elif curE2:
            if not self.is_in_cur_elem(info, stem2_pos):
                for e1 in info['elems1']:
                    hairpins = self.hairpin_count_elem2_elem1(info, curE2, e1, curJ, stem1_start, \
                                                            stem2_start, stem_length, hairpins)
                    if hairpins == -np.inf:
                        return True, hairpins
            hairpins = self.hairpin_count_elem2_elem1(info, curE2, info['curElem'], curJ, \
                                                    stem1_start, stem2_start, stem_length, \
                                                    hairpins)
        elif curE1:
            for e2 in info['elems2']:
                hairpins = self.hairpin_count_elem2_elem1(info, e2, curE1, curJ, stem1_start, \
                                                        stem2_start, stem_length, hairpins)
                if hairpins == -np.inf:
                    return True, hairpins
        if not (curE2 or info['elems2']):
            _, hairpins = self.send_to_all_check(info, curJ-1, stem1_start, stem2_start, \
                                            stem_length, hairpins, curE1_2=curE1, curE2_1=curE2)
        return True, hairpins

    def send_to_elem2_elem2(self, info, curJ, stem1_start, stem2_start, stem_length, hairpins, \
                            curE2_1="", curE2_2=""):
        if hairpins == -np.inf:
            return True, hairpins
        if not info['isKey']:
            return self.send_to_key_key(info, curJ, stem1_start, stem2_start, stem_length, \
                                        hairpins)

        stem1_pos = stem1_start + curJ
        stem2_pos = stem2_start + self.stem_out_bounds_len - 1 - curJ
        if not (self.is_in_elem2(info, stem1_pos) and self.is_in_elem2(info, stem2_pos)):
            return False, hairpins

        if curE2_1 and curE2_2:
            hairpins = self.hairpin_count_elem2_elem2(info, curE2_1, curE2_2, curJ, stem1_start,\
                                                      stem2_start, stem_length, hairpins)
        elif curE2_1:
            for e2_2 in info['elems2']:
                if self.is_in_same_elem2(info, stem1_pos, stem2_pos) and curE2_1 != e2_2:
                    continue
                hairpins = self.hairpin_count_elem2_elem2(info, curE2_1, e2_2, curJ, stem1_start,\
                                                        stem2_start, stem_length, hairpins)
                if hairpins == -np.inf:
                    return True, hairpins
        elif curE2_2:
            for e2_1 in info['elems2']:
                if self.is_in_same_elem2(info, stem1_pos, stem2_pos) and e2_1 != curE2_2:
                    continue
                hairpins = self.hairpin_count_elem2_elem2(info, e2_1, curE2_2, curJ, stem1_start,\
                                                        stem2_start, stem_length, hairpins)
                if hairpins == -np.inf:
                    return True, hairpins
        else:
            for e2_1 in info['elems2']:
                for e2_2 in info['elems2']:
                    if self.is_in_same_elem2(info, stem1_pos, stem2_pos) and e2_1 != e2_2:
                        continue
                    hairpins = self.hairpin_count_elem2_elem2(info, e2_1, e2_2, curJ, stem1_start,\
                                                stem2_start, stem_length, hairpins)
        
        if not ((curE2_1 and curE2_2) or info['elems2']):
            _, hairpins = self.send_to_all_check(info, curJ-1, stem1_start, stem2_start, \
                                                 stem_length, hairpins, \
                                                 curE2_1=curE2_1, curE2_2=curE2_2)
        return True, hairpins

    ##### Check for Hairpins #####

    def hairpin_count_elem1_elem1(self, info, elem1_1, elem1_2, curJ, stem1_start, stem2_start, \
                                  stem_length, hairpins):
        if hairpins == -np.inf:
            return hairpins
        for j in range(curJ, -1, -1):
            was_sent_to_e2_e1, hairpins = self.send_to_elem2_elem1(info, j, stem1_start, \
                                                                   stem2_start, stem_length, \
                                                                   hairpins, curE1=elem1_2)
            was_sent_to_e2_e2, hairpins = self.send_to_elem2_elem2(info, j, stem1_start, \
                                                                   stem2_start, stem_length, \
                                                                   hairpins)
            was_sent_to_e1_e2, hairpins = self.send_to_elem1_elem2(info, j, stem1_start, \
                                                                   stem2_start, stem_length, \
                                                                   hairpins, curE1=elem1_1)
            if was_sent_to_e2_e1 or was_sent_to_e2_e2 or was_sent_to_e1_e2:
                return hairpins

            stem1_start_pos = stem1_start % self.window_size
            stem2_start_pos = stem2_start % self.window_size
            

            cur_stem1_pos = (stem1_start_pos + j) % self.window_size
            cur_stem2_pos = (stem2_start_pos + self.stem_out_bounds_len - 1 - j) % self.window_size

            if cur_stem1_pos >= len(elem1_1) or cur_stem2_pos >= len(elem1_2):
                if j == 0 and stem_length > 0:
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
                continue
            
            if elem1_1[cur_stem1_pos] == converse[elem1_2[cur_stem2_pos]]:
                stem_length += 1
                if j == 0:
                    if stem_length >= self.stem_out_bounds_len:
                        hairpins = -np.inf
                        return hairpins
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
            else:
                stem_length = 0
                break
        return hairpins

    def hairpin_count_elem1_elem2(self, info, elem1, elem2, curJ, stem1_start, stem2_start, \
                                  stem_length, hairpins):
        if hairpins == -np.inf:
            return hairpins
        for j in range(curJ, -1, -1):
            was_sent_to_e2_e1, hairpins = self.send_to_elem2_elem1(info, j, stem1_start, \
                                                                stem2_start, stem_length, hairpins)
            was_sent_to_e2_e2, hairpins = self.send_to_elem2_elem2(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE2_2=elem2)
            was_sent_to_e1_e1, hairpins = self.send_to_elem1_elem1(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE1_1=elem1)
            if was_sent_to_e2_e1 or was_sent_to_e2_e2 or was_sent_to_e1_e1:
                return hairpins

            stem1_start_pos = stem1_start % self.window_size
            stem2_start_pos = stem2_start % self.window_size

            cur_stem1_pos = (stem1_start_pos + j) % self.window_size
            cur_stem2_pos = (stem2_start_pos + self.stem_out_bounds_len - 1 - j) % self.window_size

            if cur_stem1_pos >= len(elem1) or cur_stem2_pos - info['elem1Size'] >= len(elem2):
                if j == 0 and stem_length > 0:
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
                continue
            if elem1[cur_stem1_pos] == converse[elem2[cur_stem2_pos - info['elem1Size']]]:
                stem_length += 1
                if j == 0:
                    if stem_length >= self.stem_out_bounds_len:
                        hairpins = -np.inf
                        return hairpins
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
            else:
                stem_length = 0
                break
        return hairpins

    def hairpin_count_elem2_elem1(self, info, elem2, elem1, curJ, stem1_start, stem2_start, \
                                  stem_length, hairpins):
        if hairpins == -np.inf:
            return hairpins
        for j in range(curJ, -1, -1):
            was_sent_to_e1_e1, hairpins = self.send_to_elem1_elem1(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE1_2=elem1)
            was_sent_to_e1_e2, hairpins = self.send_to_elem1_elem2(info, j, stem1_start, \
                                                                stem2_start, stem_length, hairpins)
            was_sent_to_e2_e2, hairpins = self.send_to_elem2_elem2(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE2_1=elem2)
            if was_sent_to_e1_e1 or was_sent_to_e1_e2 or was_sent_to_e2_e2:
                return hairpins

            stem1_start_pos = stem1_start % self.window_size
            stem2_start_pos = stem2_start % self.window_size

            cur_stem1_pos = (stem1_start_pos + j) % self.window_size
            cur_stem2_pos = (stem2_start_pos + self.stem_out_bounds_len - 1 - j) % self.window_size

            if cur_stem2_pos >= len(elem1) or cur_stem1_pos - info['elem1Size'] >= len(elem2):
                if j == 0 and stem_length > 0:
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
                continue
            
            if elem2[cur_stem1_pos - info['elem1Size']] == converse[elem1[cur_stem2_pos]]:
                stem_length += 1
                if j == 0:
                    if stem_length >= self.stem_out_bounds_len:
                        hairpins = -np.inf
                        return hairpins
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
            else:
                stem_length = 0
                break
        return hairpins

    def hairpin_count_elem2_elem2(self, info, elem2_1, elem2_2, curJ, stem1_start, stem2_start, \
                                  stem_length, hairpins):
        if hairpins == -np.inf:
            return hairpins
        for j in range(curJ, -1, -1):
            was_sent_to_e1_e1, hairpins = self.send_to_elem1_elem1(info, j, stem1_start, \
                                                                stem2_start, stem_length, hairpins)
            was_sent_to_e1_e2, hairpins = self.send_to_elem1_elem2(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE2=elem2_2)   
            was_sent_to_e2_e1, hairpins = self.send_to_elem2_elem1(info, j, stem1_start, \
                                                                stem2_start, stem_length, \
                                                                hairpins, curE2=elem2_1)
            if was_sent_to_e1_e1 or was_sent_to_e1_e2 or was_sent_to_e2_e1:
                return hairpins

            cur_stem1_pos = (stem1_start + j) % self.window_size
            cur_stem2_pos = (stem2_start + self.stem_out_bounds_len - 1 - j) % self.window_size

            if cur_stem2_pos - info['elem1Size'] >= len(elem2_2) or \
                cur_stem1_pos - info['elem1Size'] >= len(elem2_1):
                if j == 0 and stem_length > 0:
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
                continue
            
            if elem2_1[cur_stem1_pos - info['elem1Size']] == converse[elem2_2[cur_stem2_pos - \
                                                                            info['elem1Size']]]:
                stem_length += 1
                if j == 0:
                    if stem_length >= self.stem_out_bounds_len:
                        hairpins = -np.inf
                        return hairpins
                    hairpins.append(self.calculate_hairpin_log_score(stem_length))
            else:
                stem_length = 0
                break
        return hairpins

    ##### Calculated Hairpin Log Scores #####

    def backward_hairpin_log_score(self, cur_elem, elems1, elems2, is_key=False):
        """This function calculates the log score for backward hairpins with 
        the first stem containing the last added base to the currently being constructed
        sequence, for stem lengths between 1 and the boundary defined in the stem hairpin 
        constraint, max_hairpin + 1, and for loop lengths between `loop_size_min` and 
        `loop_size_max`, which were also defined in the inputted constraints.
        
        Parameters
        ----------
        cur_elem: str
            Sequence being currently generated.
        elems1: set or list str
            List of already generated keys if the current sequence being generated is a key, 
            set of already generated payloads otherwise.
        elems2: set or list of str
            List of already generated keys if the current sequence being generated is a payload, 
            set of already generated payloads otherwise.
        is_key: bool
            True if the sequence being currently generated is a key, False otherwise.
        
        Returns
        ----------
        hairpin_log_score: float
            Log score of all backward hairpins with the first stem containing the last 
            added base to the currently being constructed sequence, for stem lengths between 
            1 and the boundary defined in the stem hairpin constraint, max_hairpin + 1, and 
            for loop lengths between `loop_size_min` and `loop_size_max`, which were also 
            defined in the inputted constraints.
        """

        hairpins = []
        hairpin_log_score = 0
        for i in range(self.stem_out_bounds_len):
            stem1_start = len(cur_elem) - 1 - i
            hairpins = self.backward_hairpin_at_pos_log_score(cur_elem, elems1, elems2, \
                                                            stem1_start, is_key=is_key)
            if hairpins == -np.inf:
                return -np.inf
            hairpin_log_score += np.sum(np.array(hairpins))

        return hairpin_log_score

    def backward_hairpin_at_pos_log_score(self, cur_elem, elems1, elems2, stem1_start, \
                                             is_key=False, loop_size_min=-1, loop_size_max=-1):
        """This function calculates the log score for backward hairpins with 
        the first stem starting at position stem1_start, and for loop lengths between 
        `loop_size_min` and `loop_size_max`, which were also defined in the inputted constraints.
        
        Parameters
        ----------
        cur_elem: str
            Sequence being currently generated.
        elems1: set or list str
            List of already generated keys if the current sequence being generated is a key, 
            set of already generated payloads otherwise.
        elems2: set or list of str
            List of already generated keys if the current sequence being generated is a payload, 
            set of already generated payloads otherwise.
        is_key: bool
            True if the sequence being currently generated is a key, False otherwise.
        
        Returns
        ----------
        hairpin_log_score: float
            Log score of all backward hairpins with the first stem starting at position stem1_start,
            and for loop lengths between `loop_size_min` and `loop_size_max`, which were also 
            defined in the inputted constraints.
        """
        hairpins = []
        if loop_size_min < 0 or loop_size_max < 0:
            loop_size_min = self.loop_size_min
            loop_size_max = self.loop_size_max

        for loopSize in range(loop_size_min, loop_size_max + 1):
            stem2_start = stem1_start - loopSize - self.stem_out_bounds_len

            # elem1 has same type (key/payload) as cur_elem
            # elem2 has opposite type (key/payload) of cur_elem
            elem1Size = self.key_size if is_key else self.payload_size
            elem2Size = self.payload_size if is_key else self.key_size

            info = {'curElem': cur_elem, 
                    'elems1': elems1, 
                    'elem1Size': elem1Size, 
                    'elems2': elems2, 
                    'elem2Size': elem2Size,
                    'isKey': is_key,
                    'firstKey': -1
                    }
            _, hairpins = self.send_to_all_check(info, self.stem_out_bounds_len - 1, stem1_start, \
                                                    stem2_start, 0, hairpins)
            if hairpins == -np.inf:
                return -np.inf

        return hairpins

    def forward_hairpin_at_pos_log_score(self, cur_elem, elems1, elems2, stem1_start, \
                                         first_key=-1, is_key=False, loop_size_min=-1, \
                                         loop_size_max=-1):
        """This function calculates the log score for forward hairpins with 
        the first stem starting at position stem1_start, and for loop lengths between 
        `loop_size_min` and `loop_size_max`, which were also defined in the inputted constraints.
        
        Parameters
        ----------
        cur_elem: str
            Sequence being currently generated.
        elems1: set or list str
            List of already generated keys if the current sequence being generated is a key, 
            set of already generated payloads otherwise.
        elems2: set or list of str
            List of already generated keys if the current sequence being generated is a payload, 
            set of already generated payloads otherwise.
        is_key: bool
            True if the sequence being currently generated is a key, False otherwise.
        
        Returns
        ----------
        hairpin_log_score: float
            Log score of all forward hairpins with the first stem starting at position stem1_start,
            and for loop lengths between `loop_size_min` and `loop_size_max`, which were also 
            defined in the inputted constraints.
        """
        hairpins = []
        if loop_size_min < 0 or loop_size_max < 0:
            loop_size_min = self.loop_size_min
            loop_size_max = self.loop_size_max

        for loopSize in range(loop_size_min, loop_size_max + 1):
            stem2_start = stem1_start + loopSize + self.stem_out_bounds_len
            
            # elem1 has same type (key/payload) as cur_elem
            # elem2 has opposite type (key/payload) of cur_elem
            elem1Size = self.key_size if is_key else self.payload_size
            elem2Size = self.payload_size if is_key else self.key_size

            info = {'curElem': cur_elem, 
                    'elems1': elems1, 
                    'elem1Size': elem1Size, 
                    'elems2': elems2, 
                    'elem2Size': elem2Size,
                    'isKey': is_key,
                    'firstKey': first_key
                    }
            _, hairpins = self.send_to_all_check(info, self.stem_out_bounds_len - 1, stem1_start, \
                                                    stem2_start, 0, hairpins)
            if hairpins == -np.inf:
                return -np.inf

        return hairpins

    def forward_hairpin_log_score(self, cur_elem, elems1, elems2, is_key=False):
        """This function calculates the log score for forward hairpins with 
        the first stem containing the last added base to the currently being constructed
        sequence, for stem lengths between 1 and the boundary defined in the stem hairpin 
        constraint, max_hairpin + 1, and for loop lengths between `loop_size_min` and 
        `loop_size_max`, which were also defined in the inputted constraints.
        
        Parameters
        ----------
        cur_elem: str
            Sequence being currently generated.
        elems1: set or list str
            List of already generated keys if the current sequence being generated is a key, 
            set of already generated payloads otherwise.
        elems2: set or list of str
            List of already generated keys if the current sequence being generated is a payload, 
            set of already generated payloads otherwise.
        is_key: bool
            True if the sequence being currently generated is a key, False otherwise.
        
        Returns
        ----------
        hairpin_log_score: float
            Log score of all forward hairpins with the first stem containing the last 
            added base to the currently being constructed sequence, for stem lengths between 
            1 and the boundary defined in the stem hairpin constraint, max_hairpin + 1, and 
            for loop lengths between `loop_size_min` and `loop_size_max`, which were also 
            defined in the inputted constraints.
        """

        hairpin_log_score = 0
        for i in range(self.stem_out_bounds_len):
            stem1_start = len(cur_elem) - 1 - i
            hairpins = self.forward_hairpin_at_pos_log_score(cur_elem, elems1, elems2, stem1_start, \
                                                                                    is_key=is_key)
            if hairpins == -np.inf:
                return -np.inf
            hairpin_log_score += np.sum(np.array(hairpins))
        return hairpin_log_score

    ##### Hairpin Validation #####

    def validate_hairpin(self, keys, payloads):
        if len(payloads) == 0 or len(keys) == 0:
            return False
        for payload in payloads:
            for i in range(self.payload_size):
                if self.forward_hairpin_at_pos_log_score(payload, payloads, keys, i) == -np.inf:
                    return False

        for key in keys:
            for i in range(self.key_size):
                if self.forward_hairpin_at_pos_log_score(key, keys,\
                                                        payloads, i, first_key=i*2, \
                                                        is_key=True) == -np.inf:
                    return False
                if self.forward_hairpin_at_pos_log_score(key, keys,\
                                                        payloads, i, first_key=i*2+1, \
                                                        is_key=True) == -np.inf:
                    return False
        return True
