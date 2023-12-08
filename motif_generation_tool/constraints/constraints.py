class Constraints:
    def __init__(self, payload_size=5, payload_num=1, max_hom=2, max_hairpin=2, loop_size=-1, \
                 min_gc=20, max_gc=60, key_size=1, key_num=1, loop_size_min=1, loop_size_max=1):
        # Payload information
        self.payload_size = payload_size
        self.payload_num = payload_num
        
        # Motifs information
        self.motif_size = payload_size + key_size * 2

        # Keys information
        self.key_size = key_size
        self.key_num = key_num

        # Constraints for keys and motifs
        max_hom = min(max_hom, 4 * key_size - 2 + 3 * payload_size)
        if key_num == 1:
            max_hom = min(max_hom, key_size + payload_size - 1)

        self.max_hom = max_hom
        self.max_hairpin = max_hairpin
        # If give loop size range:
        self.loop_size_min = loop_size_min if loop_size == -1 else loop_size
        self.loop_size_max = loop_size_max if loop_size == -1 else loop_size
        self.min_gc = min_gc
        self.max_gc = max_gc

        # Assertions
        assert(self.max_hairpin > 0)
        assert(self.max_hom > 0)
        assert(self.loop_size_min <= self.loop_size_max)
        assert(self.loop_size_min >= 0)
        assert(payload_size > 0)
        assert(key_size > 0)
        assert(key_num > 0)
        assert(payload_num > 0)
        assert(min_gc <= max_gc)
        assert(min_gc >= 0)
        assert(max_gc <= 100)
        