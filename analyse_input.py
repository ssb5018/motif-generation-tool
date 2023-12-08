from motif_generation_tool.key_payload_builder import KeyPayloadBuilder
from motif_generation_tool.constraints.constraints import Constraints
from motif_generation_tool.hyperparameters.hyperparameters import Hyperparameters
import re

async def generate_keys_and_payloads(constraints, with_constraints):
    # Get user input
    key_size = constraints['keySize']
    payload_size = constraints['payloadSize']
    key_num = constraints['keyNum']
    payload_num = constraints['payloadNum']
    max_hom = 1 if not 'maxHomopolymer' in constraints else constraints['maxHomopolymer']
    max_hairpin = 1 if not 'maxHairpin' in constraints else constraints['maxHairpin']
    loop_size_min = 1 if not 'loopMin' in constraints else constraints['loopMin']
    loop_size_max = 1 if not 'loopMax' in constraints else constraints['loopMax']
    min_gc = 25 if not 'gcContentMinPercentage' in constraints \
                else constraints['gcContentMinPercentage']
    max_gc = 65 if not 'gcContentMaxPercentage' in constraints \
                else constraints['gcContentMaxPercentage']
    
    # Verify input data is valid
    if max_hairpin <= 0 or max_hom <= 0 or loop_size_min > loop_size_max or loop_size_min < 0 or \
        payload_size <= 0 or key_size <= 0 or key_num <= 0 or payload_num <= 0 or min_gc > max_gc \
        or min_gc < 0 or max_gc > 100:
        return "", "", "", False

    constraints = Constraints(payload_size=payload_size, payload_num=payload_num, \
                             max_hom=max_hom, max_hairpin=max_hairpin, \
                             loop_size_min=loop_size_min, loop_size_max=loop_size_max,\
                             min_gc=min_gc, max_gc=max_gc, key_size=key_size, key_num=key_num)

    shapes = {'hom': 70, 'gcContent': 20, 'hairpin': 8, 'similarity': 50}
    weights = {'hom': 1, 'gcContent': 1, 'hairpin': 1, 'similarity': 1}
    hyperparams = Hyperparameters(shapes, weights)

    generate_keys_and_payloads = KeyPayloadBuilder(constraints, hyperparams)
    keys, payloads = await generate_keys_and_payloads.build_keys_and_payloads(with_constraints)
    motifs = generate_keys_and_payloads.get_motifs(keys, payloads)
    if not (keys and payloads):
        return "", "", "", False
    return format_data(keys), format_data(payloads), format_data(motifs), True

def format_data(data):
    # Reformat data to website style
    data = str(data)
    data = re.sub(r"[^ATCG ]+", "", data)
    return data
