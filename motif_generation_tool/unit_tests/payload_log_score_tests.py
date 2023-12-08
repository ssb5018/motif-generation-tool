import pytest
import numpy as np
from ..payload import payload_log_score as ls
from ..constraints import constraints as c
from ..hyperparameters import hyperparameters as h

def get_constraints(maxHairpin=2, loopSize=1, payloadSize=10, keySize=2, payloadNum=5, maxHom=1, minGC=25, maxGC=60, keyNum=5):
    payloadSize = payloadSize
    payloadNum = payloadNum
    maxHom = maxHom
    maxHairpin = maxHairpin
    loopSize = loopSize
    minGc = minGC
    maxGc = maxGC
    keySize = keySize
    keyNum = keyNum
    
    return c.Constraints(payloadSize, payloadNum, maxHom, maxHairpin, loopSize, minGc, maxGc, keySize, keyNum)

def get_hyperparameters(hairpinHyperparam=5, homHyperparam=5, gcHyperparam=5):
    hyperparams = {'hom': homHyperparam, 'gcContent': gcHyperparam, 'hairpin': hairpinHyperparam}
    return h.Hyperparameters(hyperparams)

def get_score(elem_length, elem_hyperparam, max_elem):
    if elem_length == 0:
        return 0
    return -elem_hyperparam**(elem_length/max_elem) + 1

###### GC content tests ######

def get_log_score_gc_content(gcCount, curMotifSize, minGc, maxGc, motifSize, gcHyperparam):
    weight = gcHyperparam**(curMotifSize / motifSize) - 1
    minGcContent = (100 * gcCount) / curMotifSize
    maxGcContent = (100 * gcCount) / curMotifSize
    log_score = 0
    log_score = max(log_score, weight * (minGc - minGcContent))
    log_score = max(log_score, weight * (maxGcContent - maxGc))
    return -log_score

@pytest.mark.asyncio
async def test_gc_within_bounds():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['AAGG', 'AACC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GA'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = 0
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_empty_returns_keys_gc_min():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['AAAA']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = ''
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(0, 8, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_empty_returns_keys_gc_max():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['AAGC', 'GGCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = ''
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(8, 8, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_empty_returns_keys_gc_within_bounds():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 5
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['AGTG', 'ACCT']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = ''
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = 0
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_edge_min():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 10
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['ACGA', 'TATC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'AA'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(2, 10, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_edge_max():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 10
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['ACGA', 'TGTC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GG'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(6, 10, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_within_bounds_full_payload():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 6
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['ATGA', 'TATC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'AGGAGA'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(5, 14, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_min_full_payload():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['AAAA']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GA'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = -np.inf
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_max_full_payload():
    minGC = 20
    maxGC = 60
    gcHyperparam = 5
    keySize = 4
    payloadSize = 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['GGAG', 'AAAA']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GG'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = -np.inf
    assert result == gc_log_score

@pytest.mark.asyncio
async def test_gc_min_and_max():
    minGC = 40
    maxGC = 50
    gcHyperparam = 5
    keySize = 4
    payloadSize = 7
    motifSize = payloadSize + keySize * 2
    constraints = get_constraints(minGC=minGC, maxGC=maxGC, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(gcHyperparam=gcHyperparam)
    keys = ['GGGG', 'AAAA']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'AAAAGG'
    gc_log_score = payload_log_score.motif_gc_content_log_score(curPayload)
    result = get_log_score_gc_content(2, 14, minGC, maxGC, motifSize, gcHyperparam)
    assert result == gc_log_score

###### Homopolymer tests ######

@pytest.mark.asyncio
async def test_hom_size_one():
    maxHom = 2
    homHyperparam = 5
    keySize = 4
    payloadSize = 5
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AATT', 'GGCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(1, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_isolates_long():
    maxHom = 5
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AATT', 'GGCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(4, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_end_joint():
    maxHom = 7
    homHyperparam = 5
    keySize = 4
    payloadSize = 5
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAGG', 'GGCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(6, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key_start():
    maxHom = 9
    homHyperparam = 5
    keySize = 4
    payloadSize = 5
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(8, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key_start_with_itself():
    maxHom = 10
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payloads = {'TGAAAA', 'CGAAAA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AGAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(9, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key_start_with_itself_and_other_payloads():
    maxHom = 12
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payloads = {'AAAGTA', 'AGAACA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AGAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(11, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_full_motif():
    maxHom = 20
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payloads = {'TAAATA', 'TGAACA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AAAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = -np.inf
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key():
    maxHom = 11
    homHyperparam = 5
    keySize = 6
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAAAA', 'GGCCCC']
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'AAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(10, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key_and_with_other_payloads():
    maxHom = 11
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payloads = {'TGAATA', 'TGACAA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(10, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_payload_and_part_key():
    maxHom = 10
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['TAAA', 'GGCC']
    payloads = {'TGAATA', 'TGACAA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AAAAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(9, homHyperparam, maxHom)
    assert result == hom_log_score

@pytest.mark.asyncio
async def test_hom_with_full_key_and_with_full_other_payloads():
    maxHom = 25
    homHyperparam = 5
    keySize = 4
    payloadSize = 6
    constraints = get_constraints(maxHom=maxHom, keySize=keySize, payloadSize=payloadSize)
    hyperparams = get_hyperparameters(homHyperparam=homHyperparam)
    keys = ['AAAA', 'GGCC']
    payloads = {'AAAAAA', 'TGACTT'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'AAAA'
    hom_log_score = payload_log_score.homopolymer_log_score(curPayload)
    result = get_score(24, homHyperparam, maxHom)
    assert result == hom_log_score

##### Hairpin tests ######

##### Backward hairpin tests ######

@pytest.mark.asyncio
async def test_stem1_in_payload_stem2_in_key_edge():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GA'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_empty_curPayload_returns_zero():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = ''
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload_stem2_in_payload_whole():
    maxHairpin = 3
    loopSize = 2
    payloadSize = 4
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TT', 'AC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GATC'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_full_hairpin_stem1_in_payload_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload_stem2_in_payload_and_key_with_2_keys_danger():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'AAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 4 + get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload_stem2_in_key_and_payload_part_hairpin_with_2_keys_danger():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'ATT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(1, hairpinHyperparam, maxHairpin) * 4
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload1_stem2_in_payload1_whole():
    maxHairpin = 1
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'ATTAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_full_hairpin_in_payload_at_edge():
    maxHairpin = 1
    loopSize = 1
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'ATTAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload1_stem2_in_payload2_part_in_key():
    maxHairpin = 2
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = {'ATCGG', 'ATCCG'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'CT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(1, hairpinHyperparam, maxHairpin) * (len(payloads) + 1) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload1_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AAGT', 'ATGC']
    payloads = {'TATAAGGA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'CT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_stem1_in_payload1_stem2_in_payload1_and_payload2(): # impossible
    maxHairpin = 4
    loopSize = 1
    payloadSize = 7
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'ATAATAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

###### Forward hairpin tests ######

@pytest.mark.asyncio
async def test_forward_empty_curPayload_returns_zero():
    maxHairpin = 2
    loopSize = 1
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = ''
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(0, hairpinHyperparam, maxHairpin)
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload_stem2_in_payload_whole():
    maxHairpin = 4
    loopSize = 2
    payloadSize = 4
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TT', 'AC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GATC'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_full_hairpin_stem1_in_payload_stem2_in_key():
    maxHairpin = 2
    loopSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'GAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = get_score(2, hairpinHyperparam, maxHairpin) * 2
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload_stem2_in_payload_and_key():
    maxHairpin = 2
    loopSize = 8
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AA', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'TAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload_stem2_in_payload_and_key_whole():
    maxHairpin = 1
    loopSize = 6 + 12 * 2
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TG', 'GC']
    payloads = {'AATCATCGAA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'TAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_full_hairpin_in_payload_at_edge():
    maxHairpin = 1
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'ATTAT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload1_and_key_stem2_in_payload2():
    maxHairpin = 2
    loopSize = 2
    payloadSize = 5
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'GC']
    payloads = {'TTAGA'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'TTTCT'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload1_stem2_in_key():
    maxHairpin = 1
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 4
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AAGT', 'ATAC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'TATAAGTA'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload1_and_key_stem2_in_key():
    maxHairpin = 2
    loopSize = 1
    payloadSize = 8
    hairpinHyperparam = 5
    keySize = 6
    constraints = get_constraints(maxHairpin, loopSize, payloadSize, keySize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['TAATCT', 'TAGGGC']
    payloads = set()
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    curPayload = 'TATAAGGA'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore

@pytest.mark.asyncio
async def test_forward_stem1_in_payload1_stem2_in_payload12_and_payload2():
    maxHairpin = 3
    loopSize = 8
    payloadSize = 7
    hairpinHyperparam = 5
    constraints = get_constraints(maxHairpin, loopSize, payloadSize)
    hyperparams = get_hyperparameters(hairpinHyperparam=hairpinHyperparam)
    keys = ['AT', 'CC']
    payloads = {'CCATCGG'}
    payload_log_score = ls.PayloadLogScore(constraints, hyperparams)
    await payload_log_score.add_keys(keys)
    await payload_log_score.add_payloads(payloads)
    curPayload = 'CTAGATC'
    hairpinScore = payload_log_score.hairpin_log_score(curPayload)
    result = -np.inf
    assert result == hairpinScore
