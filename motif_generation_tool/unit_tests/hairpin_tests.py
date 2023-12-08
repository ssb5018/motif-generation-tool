import pytest
import numpy as np
from ..constraints import hairpin as ls
from ..constraints import constraints as c
from ..hyperparameters import hyperparameters as h

def get_constraints(maxHairpin=2, loopSize=-1, payloadSize=10, keySize=2, payloadNum=5, maxHom=1, minGC=25, maxGC=60, loopSizeMin=1, loopSizeMax=1):
    payloadSize = payloadSize
    payloadNum = payloadNum
    maxHom = 1
    maxHairpin = maxHairpin
    loopSize = loopSize
    loopSizeMin = loopSizeMin
    loopSizeMax = loopSizeMax
    minGc = 25
    maxGc = 60
    keySize = keySize
    keyNum = 10
    
    return c.Constraints(payloadSize, payloadNum, maxHom, maxHairpin, loopSize, minGc, maxGc, keySize, keyNum, loopSizeMin, loopSizeMax)

def get_hyperparameters(hairpinHyperparam=5):
    hyperparams = {'hairpin': hairpinHyperparam}
    return h.Hyperparameters(hyperparams)

def get_score(elem_length, elem_hyperparam, max_elem):
    if elem_length == 0:
        return 0
    return -elem_hyperparam**(elem_length/max_elem) + 1

##### Helper Functions tests #####

def test_is_in_cur_elem():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_cur_elem(info, 9)
    result = True
    assert result == hairpinScore

def test_is_in_cur_elem_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_cur_elem(info, 10)
    result = False
    assert result == hairpinScore

def test_is_in_cur_elem_negative_pos_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_cur_elem(info, -1)
    result = False
    assert result == hairpinScore

def test_is_in_elem2():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_elem2(info, 22)
    result = True
    assert result == hairpinScore

def test_is_in_elem2_negative_pos():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_elem2(info, -25)
    result = True
    assert result == hairpinScore

def test_is_in_elem2_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_elem2(info, -15)
    result = False
    assert result == hairpinScore

def test_is_in_other_elem1():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_other_elem1(info, 12)
    result = True
    assert result == hairpinScore

def test_is_in_other_elem1_negative():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_other_elem1(info, -15)
    result = True
    assert result == hairpinScore

def test_is_in_other_elem1_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_other_elem1(info, 9)
    result = False
    assert result == hairpinScore

def test_is_in_same_elem1_cur_elem():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem1(info, 3, 9)
    result = True
    assert result == hairpinScore

def test_is_in_same_elem1_negative_pos():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem1(info, -15, -18)
    result = True
    assert result == hairpinScore

def test_is_in_same_elem1_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem1(info, -15, 9)
    result = False
    assert result == hairpinScore

def test_is_in_same_elem2():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem2(info, 22, 23)
    result = True
    assert result == hairpinScore

def test_is_in_same_elem2_is_key_true():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_key = 'G'
    info = {'cur_elem': cur_key, 
            'elems1': keys, 
            'elem1Size': 2, 
            'elems2': payloads, 
            'elem2Size': 10,
            'isKey': True,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem2(info, 14, 17)
    result = True
    assert result == hairpinScore

def test_is_in_same_elem2_negative_pos():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem2(info, -13, -14)
    result = True
    assert result == hairpinScore

def test_is_in_same_elem2_false():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': -1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.is_in_same_elem2(info, -12, -14)
    result = False
    assert result == hairpinScore

def test_get_key_at_pos():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.get_key_at_pos(info, 23)
    result = 'GC'
    assert result == hairpinScore

def test_get_key_at_negative_pos():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.get_key_at_pos(info, -1)
    result = 'AT'
    assert result == hairpinScore

def test_stem_contains_only_one_key():
    maxHairpin = 16
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, 5)
    result = True
    assert result == hairpinScore

def test_stem_contains_only_one_key_false():
    maxHairpin = 17
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, 5)
    result = False
    assert result == hairpinScore

def test_stem_contains_only_one_key_false_negative_pos():
    maxHairpin = 18
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, -8)
    result = False
    assert result == hairpinScore

def test_stem_contains_only_one_key_negative_pos():
    maxHairpin = 17
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, -8)
    result = True
    assert result == hairpinScore

def test_stem_contains_only_one_key_negative_pos_start_elem2_false():
    maxHairpin = 11
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, -13)
    result = False
    assert result == hairpinScore

def test_stem_contains_only_one_key_negative_pos_start_elem2_is_key_false():
    maxHairpin = 11
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': True,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_only_one_key(info, -11)
    result = False
    assert result == hairpinScore

def test_stem_contains_no_key():
    maxHairpin = 5
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_no_key(info, -8)
    result = True
    assert result == hairpinScore

def test_stem_contains_no_key_false():
    maxHairpin = 6
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 1
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stem_contains_no_key(info, 4)
    result = False
    assert result == hairpinScore

def test_stems_in_same_key_even_after_shift():
    maxHairpin = 5
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 0
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stems_in_same_key_even_after_shift(info, -1)
    result = True
    assert result == hairpinScore

def test_stems_in_same_key_even_after_shift_false():
    maxHairpin = 15
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    cur_payload = 'GA'
    info = {'cur_elem': cur_payload, 
            'elems1': payloads, 
            'elem1Size': 10, 
            'elems2': keys, 
            'elem2Size': 2,
            'isKey': False,
            'firstKey': 0
            }
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    hairpinScore = hairpin_log_score.stems_in_same_key_even_after_shift(info, -1)
    result = False
    assert result == hairpinScore

##### Payload Hairpin Log Score tests ######

##### Backward hairpin tests ######

def test_stem1_in_payload_stem2_in_key_edge():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    cur_payload = 'GA'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_stem1_in_payload_stem2_non_existent():
    maxHairpin = 2
    loopSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'GA'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_empty_curPayload_returns_zero():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = ''
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_empty_key_hairpin_between_payloads():
    maxHairpin = 3
    loopSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = []
    payloads = {'AAAAAAAAAA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'TTT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) + get_score(2, hairpinHyperparam, maxHairpin) + get_score(3, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_stem1_in_payload_stem2_in_payload_whole():
    maxHairpin = 3
    loopSize = 2
    payloadSize = 4
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TT', 'AC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'GATC'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_full_hairpin_stem1_in_payload_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'GAT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_stem1_in_payload_stem2_in_payload_and_key_with_2_keys_danger():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'AAT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 4 + get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_stem1_in_payload_stem2_in_key_and_payload_part_hairpin_with_2_keys_danger():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 4
    assert result == hairpinScore

def test_stem1_in_payload1_stem2_in_payload1_whole():
    maxHairpin = 1
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATTAT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_full_hairpin_in_payload_at_edge():
    maxHairpin = 1
    loopSize = 1
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATTAT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_stem1_in_payload1_stem2_in_payload2_part_in_key():
    maxHairpin = 2
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = {'ATCGG', 'ATCCG'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) * (len(payloads) + 1) * 2
    assert result == hairpinScore

def test_stem1_in_payload1_and_key_stem2_in_payload2():
    maxHairpin = 3
    loopSize = 2
    payloadSize = 8
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = {'TATAAGGA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(3, hairpinHyperparam, maxHairpin) * 2 + get_score(1, hairpinHyperparam, maxHairpin) * 4
    assert result == hairpinScore

def test_stem1_in_payload1_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AAGT', 'ATGC']
    payloads = {'TATAAGGA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_stem1_in_payload1_and_key_stem2_in_key():
    maxHairpin = 3
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TAGATT', 'TAGGGC']
    payloads = {'TATAAGGA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(3, hairpinHyperparam, maxHairpin) * 4
    assert result == hairpinScore

def test_stem1_in_payload1_stem2_in_payload1_and_payload2(): # impossible
    maxHairpin = 4
    loopSize = 1
    payloadSize = 7
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATAATAT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_stem1_in_payload1_stem2_in_payload1_and_payload2():
    maxHairpin = 4
    loopSize = 1
    payloadSize = 7
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AA', 'GC']
    payloads = {'TCATCGA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'TGATTT'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(4, hairpinHyperparam, maxHairpin) * 2 + get_score(3, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

###### Forward hairpin tests ######

def test_forward_empty_curPayload_returns_zero():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = ''
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_stem1_in_payload_stem2_in_payload_whole():
    maxHairpin = 4
    loopSize = 2
    payloadSize = 4
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TT', 'AC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'GATC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_full_hairpin_stem1_in_payload_stem2_in_key():
    maxHairpin = 2
    loopSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'GAT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_forward_stem1_in_payload_stem2_in_payload_and_key():
    maxHairpin = 2
    loopSize = 8
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AA', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'TAT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload_stem2_in_payload_and_key_whole():
    maxHairpin = 1
    loopSize = 6 + 12 * 2
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TG', 'GC']
    payloads = {'AATCATCGAA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'TAT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_stem2_in_payload1():
    maxHairpin = 1
    loopSize = 3
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATTAT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

def test_forward_full_hairpin_in_payload_at_edge():
    maxHairpin = 1
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'ATTAT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_and_key_stem2_in_payload2():
    maxHairpin = 2
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = {'TTAGA'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'TTTCT'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_stem2_in_key():
    maxHairpin = 1
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AAGT', 'ATAC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'TATAAGTA'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_and_key_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TAATCT', 'TAGGGC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    cur_payload = 'TATAAGGA'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_stem2_in_payload12_and_payload2():
    maxHairpin = 3
    loopSize = 8
    payloadSize = 7
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC']
    payloads = {'CCATCGG'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CTAGATC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_in_payload1_stem2_in_payload12_and_payload2_order_key_matters_multiple_hairpins():
    maxHairpin = 4
    loopSize = 22
    payloadSize = 3
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC', 'TT']
    payloads = {'ACG'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CG'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(4, hairpinHyperparam, maxHairpin) * 2 + get_score(3, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_stem1_in_payload1_stem2_in_payload12_and_payload2_order_key_matters():
    maxHairpin = 4
    loopSize = 22 + 5
    payloadSize = 3
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC', 'TT']
    payloads = {'ACG'}
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    
    cur_payload = 'CG'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(cur_payload, payloads, keys)
    result = get_score(4, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

###### Key Hairpin Log Score tests ######

###### Backward hairpin tests ######

def test_stem1_and_stem2_in_key1_edge_reached_maxHairpin():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = -np.inf
    assert result == hairpinScore

def test_stem1_and_stem2_in_key1_edge():
    maxHairpin = 3
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(2, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 11
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC', 'AG']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_part_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'AG', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'G'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_empty_stem1_returns_zero():
    maxHairpin = 2
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AG', 'GC']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = ''
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_stem1_in_key1_and_stem2_in_key2_whole():
    maxHairpin = 6
    loopSize = 16
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, keySize=6)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['ATATAT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GCGCGC'
    hairpinScore = hairpin_log_score.backward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 2 + get_score(2, hairpinHyperparam, maxHairpin) * 2 + get_score(3, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

###### Forward hairpin tests ######

def test_forward_stem1_and_stem2_in_key1_edge_reached_maxHairpin():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = -np.inf
    assert result == hairpinScore

def test_forward_stem1_and_stem2_in_key1_edge():
    maxHairpin = 2
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(2, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 3
    loopSize = 11
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AG', 'TT', 'AT']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'GC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_part_stem1_in_key1_and_stem2_in_key2_part_edge():
    maxHairpin = 2
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['GC', 'AG']
    payloads = set()
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    
    curKey = 'G'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_part_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 8
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'GC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_part_stem1_and_stem2_in_key1_part():
    maxHairpin = 1
    loopSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'GAAC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_part_stem1_and_full_stem2_in_key1_part():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'AAGAAC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

def test_forward_full_stem1_and_full_stem2_in_key1():
    maxHairpin = 1
    loopSize = 12
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'AAGTAC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = -np.inf
    assert result == hairpinScore

def test_forward_full_stem1_and_part_stem2_in_key1():
    maxHairpin = 1
    loopSize = 9
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    hairpin_log_score = ls.Hairpin(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'TAGTAC'
    hairpinScore = hairpin_log_score.forward_hairpin_log_score(curKey, keys, payloads, is_key=True)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

###### Hairpin Validation Tests ######

def test_valid_hairpin():
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin)
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CA', 'GG', 'AC']
    payloads = {'CCACACACAC'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = True
    assert result == valid

def test_valid_hairpin_false_key_payload():
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, )
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CA', 'GG', 'AC']
    payloads = {'CGACACACAC'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid

def test_valid_hairpin_false_in_payload():
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, )
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CA', 'GG', 'AC']
    payloads = {'CCACGCAGCA'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid

def test_valid_hairpin_false_in_key():
    keySize = 5
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, keySize=keySize)
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CGACG', 'AAACA', 'AAAAA']
    payloads = {'CCACACAGCA'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid

def test_valid_hairpin_false_same_key():
    keySize = 5
    loopSizeMin = 13
    loopSizeMax = 13
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, keySize=keySize, loopSizeMin=loopSizeMin, loopSizeMax=loopSizeMax)
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CGACG', 'AACAA', 'AAAAA']
    payloads = {'CCACACAGCA'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid

def test_valid_hairpin_false_order_key_matters():
    keySize = 5
    loopSizeMin = 44
    loopSizeMax = 44
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, keySize=keySize, loopSizeMin=loopSizeMin, loopSizeMax=loopSizeMax)
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CGAAG', 'AACAA', 'ACGAA']
    payloads = {'CCACACAGCA'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid

def test_valid_hairpin_false_in_different_payloads():
    keySize = 5
    loopSizeMin = 12
    loopSizeMax = 12
    maxHairpin = 1
    constraints = get_constraints(maxHairpin=maxHairpin, keySize=keySize, loopSizeMin=loopSizeMin, loopSizeMax=loopSizeMax)
    hairpin_validation = ls.Hairpin(constraints)
    keys = ['CGAAG', 'AACAA', 'ACGAA']
    payloads = {'CCACACAGCA', 'CCCGACAGCA', 'CCACGCAGCA'}
    valid = hairpin_validation.validate_hairpin(keys, payloads)
    result = False
    assert result == valid
