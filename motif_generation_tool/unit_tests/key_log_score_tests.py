import pytest
import numpy as np
from ..key import key_log_score as ls
from ..constraints import constraints as c
from ..hyperparameters import hyperparameters as h


def get_constraints(maxHairpin=2, loopSize=1, payloadSize=10, keySize=2, maxHom=1, minGc=25, maxGc=60):
    payloadSize = payloadSize
    payloadNum = 5
    maxHom = maxHom
    maxHairpin = maxHairpin
    loopSize = loopSize
    minGc = minGc
    maxGc = maxGc
    keySize = keySize
    
    return c.Constraints(payloadSize, payloadNum, maxHom, maxHairpin, loopSize, minGc, maxGc, keySize)

def get_hyperparameters(hairpinHyperparam=5, homHyperparam=5, gcHyperparam=5):
    hyperparams = {'hom': homHyperparam, 'gcContent': gcHyperparam, 'hairpin': hairpinHyperparam}
    return h.Hyperparameters(hyperparams)

def get_score(elem_length, elem_hyperparam, max_elem):
    if elem_length == 0:
        return 0
    return -elem_hyperparam**(elem_length/max_elem) + 1

###### GC content tests ######

def get_score_gc_content(gcCount, curSize, minGc, maxGc, totalSize, gcHyperparam):
    weight = gcHyperparam**(curSize / totalSize) - 1
    minGcContent = (100 * gcCount) / curSize
    maxGcContent = (100 * gcCount) / curSize
    log_score = 0
    log_score = max(log_score, weight * (minGc - minGcContent)) # instead of minGC, do with average (center)?
    log_score = max(log_score, weight * (maxGcContent - maxGc)) # instead of maxGC, do with average (center)?
    return -log_score

###### GC content within key tests ######

@pytest.mark.asyncio
async def test_gc_within_key_within_bounds():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    motifSize = 2 * keySize + payloadSize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGTG', 'GACT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'G'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'A')
    result = get_score_gc_content(2, 4, minGc, maxGc, motifSize, gcHyperparam)
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_key_min():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    motifSize = 2 * keySize + payloadSize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGTG', 'GACT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'A'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'A')
    result = get_score_gc_content(0, 4, minGc, maxGc, motifSize, gcHyperparam)
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_key_min_edge():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 12
    payloadSize = 5
    motifSize = 2 * keySize + payloadSize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'GGGGGGAAAAAA', 'CCCCCCAAAAAA'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'AAACAAAAA'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'G')
    result = get_score_gc_content(4, 20, minGc, maxGc, motifSize, gcHyperparam)
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_key_max():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    motifSize = 2 * keySize + payloadSize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGTG', 'GACT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'G'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'C')
    result = get_score_gc_content(4, 4, minGc, maxGc, motifSize, gcHyperparam)
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_key_max_edge():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 12
    payloadSize = 5
    motifSize = 2 * keySize + payloadSize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGAGAGTGTGTG', 'GAGAGACTCTCT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'CAGCAAGCA'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'G')
    result = get_score_gc_content(12, 20, minGc, maxGc, motifSize, gcHyperparam)
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_motif_went_over_gc_count_limit():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 1
    motifSize = payloadSize + keySize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGTG', 'GACT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'CG'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'C')
    result = -np.inf
    assert result == gcScore

@pytest.mark.asyncio
async def test_gc_within_motif_went_under_gc_count_limit():
    minGc = 20
    maxGc = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 1
    motifSize = payloadSize + keySize
    constraints = get_constraints(minGc=minGc, maxGc=maxGc, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = {'AGTG', 'GACT'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'AAA'
    for b in curKey:
        await key_log_score.add_base(b)
    gcScore = key_log_score.key_gc_content_log_score(curKey + 'A')
    result = -np.inf
    assert result == gcScore

###### Homopoylmer tests ######

@pytest.mark.asyncio
async def test_hom_empty_key_returns_zero():
    maxHom = 2
    homHyperparam = 5
    constraints = get_constraints(maxHom=maxHom)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    curKey = ''
    homScore = key_log_score.homopolymer_log_score(curKey)
    result = get_score(0, homHyperparam, maxHom)
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_size_one():
    maxHom = 2
    homHyperparam = 5
    constraints = get_constraints(maxHom=maxHom)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = {'AT', 'GC'}
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'G'
    homScore = key_log_score.homopolymer_log_score(curKey)
    result = get_score(1, homHyperparam, maxHom)
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_full():
    maxHom = 10
    homHyperparam = 5
    keySize = 10
    constraints = get_constraints(maxHom=maxHom, keySize=keySize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    curKey = 'GGGGGGGGG'
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'G')
    result = get_score(10, homHyperparam, maxHom)
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_part():
    maxHom = 6
    homHyperparam = 5
    keySize = 10
    constraints = get_constraints(maxHom=maxHom, keySize=keySize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    curKey = 'GGGGTGGGG'
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'G')
    result = get_score(5, homHyperparam, maxHom)
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_empty_returns_zero():
    maxHom = 2
    homHyperparam = 5
    keySize = 10
    constraints = get_constraints(maxHom=maxHom, keySize=keySize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    curKey = ''
    homScore = key_log_score.homopolymer_log_score(curKey)
    result = 0
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_within_motif_end_and_start_keys_hom_maxHom1_end():
    maxHom = 1
    homHyperparam = 5
    keySize = 3
    constraints = get_constraints(maxHom=maxHom, keySize=keySize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['GAT', 'GTA', 'GTC']
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'AT'
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'G')
    result = -np.inf
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_within_motif_end_and_start_keys_hom_maxHom1_start():
    maxHom = 1
    homHyperparam = 5
    keySize = 3
    constraints = get_constraints(maxHom=maxHom, keySize=keySize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['GAT', 'CTG', 'TGC', 'TAG']
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = ''
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'A')
    result = -np.inf
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_within_motif_end_and_start_keys_hom_maxHom2_payloadSize1_start():
    maxHom = 1
    homHyperparam = 5
    keySize = 3
    payloadSize = 3
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['GAT', 'CGA', 'TGC', 'TAG']
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = ''
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'A')
    result = -np.inf
    assert result == homScore

@pytest.mark.asyncio
async def test_hom_within_motif_end_and_start_keys_hom_maxHom2_payloadSize1_end():
    maxHom = 1
    homHyperparam = 5
    keySize = 3
    payloadSize = 3
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['GAT', 'CGA', 'TGC', 'TAC']
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'AT'
    key_log_score.start_new_key()
    for b in curKey:
        await key_log_score.add_base(b)
    homScore = key_log_score.homopolymer_log_score(curKey + 'G')
    result = -np.inf
    assert result == homScore

###### Hairpin tests ######

###### Backward hairpin tests ######

@pytest.mark.asyncio
async def test_stem1_and_stem2_in_key1_edge_reached_maxHairpin():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'GC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 11
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC', 'AG']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'GC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_part_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'AG', 'GC']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'G'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_empty_stem1_returns_zero():
    maxHairpin = 2
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AG', 'GC']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = ''
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_key1_and_stem2_in_key2_whole():
    maxHairpin = 6
    loopSize = 16
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, keySize=6)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['ATATAT']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'GCGCGC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 2 + get_score(2, hairpinHyperparam, maxHairpin) * 2 + get_score(3, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

###### Forward hairpin tests ######

@pytest.mark.asyncio
async def test_forward_stem1_and_stem2_in_key1_edge_reached_maxHairpin():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'GC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 3
    loopSize = 11
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AG', 'TT', 'AT']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'GC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_part_stem1_in_key1_and_stem2_in_key2_part_edge():
    maxHairpin = 2
    loopSize = 10
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['GC', 'AG']
    payloads = set()
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    await key_log_score.add_keys(keys)
    curKey = 'G'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_part_stem1_and_stem2_in_key1_part_edge():
    maxHairpin = 1
    loopSize = 8
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'GC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_part_stem1_and_stem2_in_key1_part():
    maxHairpin = 1
    loopSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'GAAC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_part_stem1_and_full_stem2_in_key1_part():
    maxHairpin = 1
    loopSize = 10
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'AAGAAC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_full_stem1_and_full_stem2_in_key1():
    maxHairpin = 1
    loopSize = 12
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'AAGTAC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_full_stem1_and_part_stem2_in_key1():
    maxHairpin = 1
    loopSize = 9
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, keySize=keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    key_log_score = ls.KeyLogScore(constraints, hyperparams)
    keys = []
    payloads = set()
    curKey = 'TAGTAC'
    hairpinScore = key_log_score.hairpin_log_score(curKey)
    result = get_score(1, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

