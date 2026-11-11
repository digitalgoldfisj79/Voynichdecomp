#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import unittest

from codec import (
    VERSION,
    BitReader,
    BitWriter,
    CodecError,
    cost_model,
    deserialize,
    elias_delta_length,
    explicit_latent_path_codelength,
    kt_codelength,
    partition_codelength,
    serialize,
    serialized_bit_length,
    state_topology_codelength,
    write_elias_delta,
)


class CanonicalCodecTests(unittest.TestCase):
    def roundtrip(self, value):
        payload, bit_length = serialize(value)
        self.assertEqual(deserialize(payload, bit_length), value)
        return payload, bit_length

    def test_roundtrips(self):
        examples = [
            None,
            False,
            True,
            0,
            1,
            -1,
            123456,
            "",
            "Voynich",
            "é",
            [],
            [1, "a", None],
            {},
            {"z": 1, "a": [True, -3]},
        ]
        for example in examples:
            with self.subTest(example=example):
                self.roundtrip(example)

    def test_dictionary_order_is_canonical(self):
        first = {"z": 1, "a": 2, "ä": 3}
        second = {"ä": 3, "a": 2, "z": 1}
        self.assertEqual(serialize(first), serialize(second))

    def test_serialization_is_injective_on_conformance_set(self):
        values = [None, False, True, 0, 1, -1, "", "0", [], [0], {}, {"a": 0}]
        encodings = {serialize(value) for value in values}
        self.assertEqual(len(encodings), len(values))

    def test_no_complete_encoding_prefixes_another(self):
        values = [None, False, True, 0, 1, -1, "", "a", [], [0], {}, {"a": 0}]
        bitstrings = []
        for value in values:
            payload, length = serialize(value)
            bits = "".join(str((payload[i // 8] >> (7 - i % 8)) & 1) for i in range(length))
            bitstrings.append(bits)
        for i, left in enumerate(bitstrings):
            for j, right in enumerate(bitstrings):
                if i != j:
                    self.assertFalse(right.startswith(left), (values[i], values[j]))

    def test_rejects_floats_and_non_string_keys(self):
        with self.assertRaises(CodecError):
            serialize(1.25)
        with self.assertRaises(CodecError):
            serialize({1: "x"})

    def test_trailing_bits_rejected(self):
        payload, length = serialize(1)
        with self.assertRaises(CodecError):
            deserialize(payload + b"\x00", length + 1)


class IntegerCodeTests(unittest.TestCase):
    def test_known_elias_delta_lengths(self):
        expected = {1: 1, 2: 4, 3: 4, 4: 5, 7: 5, 8: 8, 15: 8, 16: 9}
        self.assertEqual({n: elias_delta_length(n) for n in expected}, expected)

    def test_delta_roundtrip(self):
        for n in range(1, 1000):
            writer = BitWriter()
            write_elias_delta(writer, n)
            payload, length = writer.to_bytes()
            reader = BitReader(payload, length)
            from codec import read_elias_delta
            self.assertEqual(read_elias_delta(reader), n)
            self.assertTrue(reader.exhausted)


class UniversalCostTests(unittest.TestCase):
    def test_kt_empty_sequence_cost_zero(self):
        self.assertAlmostEqual(kt_codelength([0, 0]), 0.0, places=12)

    def test_kt_binary_one_each_is_three_bits(self):
        self.assertAlmostEqual(kt_codelength([1, 1]), 3.0, places=12)

    def test_kt_is_label_permutation_invariant(self):
        self.assertAlmostEqual(kt_codelength([9, 3, 1]), kt_codelength([1, 9, 3]), places=12)

    def test_zero_categories_cost_information(self):
        self.assertGreater(kt_codelength([10, 0, 0]), kt_codelength([10, 0]))

    def test_partition_cost_increases_when_membership_is_less_concentrated(self):
        self.assertGreater(partition_codelength([2, 2]), partition_codelength([3, 1]))

    def test_partition_invariant_to_class_order(self):
        self.assertAlmostEqual(partition_codelength([2, 4, 1]), partition_codelength([1, 2, 4]))

    def test_topology_cost(self):
        one = state_topology_codelength(2, [1, 1])
        two = state_topology_codelength(3, [2, 2, 2])
        self.assertGreater(two, one)

    def test_latent_path_is_not_free(self):
        no_transitions = explicit_latent_path_codelength([0], 2, [])
        longer = explicit_latent_path_codelength([0, 1, 0, 1], 2, [])
        reset = explicit_latent_path_codelength([0, 1, 0, 1], 2, [2])
        self.assertGreater(no_transitions, 0)
        self.assertGreater(longer, no_transitions)
        self.assertNotEqual(reset, longer)


class ModelCostTests(unittest.TestCase):
    def model(self):
        return {
            "codec_version": VERSION,
            "name": "conformance-model",
            "surface_inventory": ["daiin", "ol", "chedy", "qokedy"],
            "plaintext_units": ["A", "B"],
            "class_sizes": [2, 2],
            "num_states": 2,
            "outdegrees": [2, 1],
            "transition_counts": [[8, 2], [1, 9]],
            "emission_counts": [[5, 5, 0, 0], [0, 0, 7, 3]],
            "latent_path_mode": "explicit",
            "latent_path": [0, 0, 1, 1, 0],
            "reset_points": [3],
            "external_model_count": 3,
            "external_model_index": 1,
        }

    def test_cost_report_is_deterministic(self):
        first = cost_model(self.model()).as_dict()
        model = json.loads(json.dumps(self.model(), sort_keys=True))
        second = cost_model(model).as_dict()
        self.assertEqual(first, second)

    def test_extra_operational_material_changes_h_full_only(self):
        base = self.model()
        expanded = dict(base)
        expanded["historical_instruction"] = "At every line boundary reset the wheel."
        a = cost_model(base)
        b = cost_model(expanded)
        self.assertGreater(b.canonical_serialization_bits, a.canonical_serialization_bits)
        self.assertAlmostEqual(b.structural_universal_bits, a.structural_universal_bits)

    def test_invalid_external_model_choice_rejected(self):
        model = self.model()
        model["external_model_index"] = 3
        with self.assertRaises(CodecError):
            cost_model(model)

    def test_marginalized_path_requires_valid_log_probability(self):
        model = self.model()
        model["latent_path_mode"] = "marginalized"
        model.pop("latent_path")
        model.pop("reset_points")
        model["marginal_log2_probability"] = -123.25
        report = cost_model(model)
        self.assertAlmostEqual(report.latent_path_bits, 123.25)
        model["marginal_log2_probability"] = 1.0
        with self.assertRaises(CodecError):
            cost_model(model)

    def test_expected_conformance_vector(self):
        report = cost_model(self.model())
        # Frozen after independent execution. A change requires a version bump.
        expected = {
            "canonical_serialization_bits": 2634,
            "partition_bits": 7.584962500721156,
            "topology_bits": 11.169925001442312,
            "transition_kt_bits": 18.25055702456491,
            "emission_kt_bits": 25.55759498287183,
            "latent_path_bits": 13.415037499278844,
            "external_model_index_bits": 1.584962500721156,
            "structural_universal_bits": 77.56303950860021,
        }
        self.assertEqual(report.canonical_serialization_bits, expected["canonical_serialization_bits"])
        for key, value in expected.items():
            if key != "canonical_serialization_bits":
                self.assertAlmostEqual(report.as_dict()[key], value, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
