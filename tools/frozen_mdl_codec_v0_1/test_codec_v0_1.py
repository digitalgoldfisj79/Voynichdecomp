#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from codec_v0_1 import (
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
    read_elias_delta,
    serialize,
    state_topology_codelength,
    write_elias_delta,
)


class CanonicalCodecTests(unittest.TestCase):
    def test_roundtrips(self):
        examples = [
            None, False, True, 0, 1, -1, 123456, "", "Voynich", "é",
            [], [1, "a", None], {}, {"z": 1, "a": [True, -3]},
        ]
        for value in examples:
            with self.subTest(value=value):
                payload, length = serialize(value)
                self.assertEqual(deserialize(payload, length), value)

    def test_dictionary_order_is_canonical(self):
        self.assertEqual(
            serialize({"z": 1, "a": 2, "ä": 3}),
            serialize({"ä": 3, "a": 2, "z": 1}),
        )

    def test_serialization_is_injective_on_conformance_set(self):
        values = [None, False, True, 0, 1, -1, "", "0", [], [0], {}, {"a": 0}]
        self.assertEqual(len({serialize(value) for value in values}), len(values))

    def test_no_complete_encoding_prefixes_another(self):
        values = [None, False, True, 0, 1, -1, "", "a", [], [0], {}, {"a": 0}]
        strings = []
        for value in values:
            payload, length = serialize(value)
            strings.append("".join(str((payload[i // 8] >> (7 - i % 8)) & 1) for i in range(length)))
        for i, left in enumerate(strings):
            for j, right in enumerate(strings):
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
            payload, length = writer.finish()
            reader = BitReader(payload, length)
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

    def test_partition_is_class_order_invariant(self):
        self.assertAlmostEqual(partition_codelength([2, 4, 1]), partition_codelength([1, 2, 4]))

    def test_topology_cost_increases(self):
        self.assertGreater(state_topology_codelength(3, [2, 2, 2]), state_topology_codelength(2, [1, 1]))

    def test_latent_path_is_not_free(self):
        one = explicit_latent_path_codelength([0], 2, [])
        longer = explicit_latent_path_codelength([0, 1, 0, 1], 2, [])
        reset = explicit_latent_path_codelength([0, 1, 0, 1], 2, [2])
        self.assertGreater(one, 0)
        self.assertGreater(longer, one)
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
        second = cost_model(json.loads(json.dumps(self.model(), sort_keys=True))).as_dict()
        self.assertEqual(first, second)

    def test_extra_operational_material_changes_h_full_only(self):
        base = self.model()
        expanded = dict(base)
        expanded["historical_instruction"] = "At every line boundary reset the wheel."
        a, b = cost_model(base), cost_model(expanded)
        self.assertGreater(b.canonical_serialization_bits, a.canonical_serialization_bits)
        self.assertAlmostEqual(b.structural_universal_bits, a.structural_universal_bits)

    def test_invalid_external_model_choice_rejected(self):
        model = self.model()
        model["external_model_index"] = 3
        with self.assertRaises(CodecError):
            cost_model(model)

    def test_marginalized_path_requires_exact_rational(self):
        model = self.model()
        model["latent_path_mode"] = "marginalized"
        model.pop("latent_path")
        model.pop("reset_points")
        model["marginal_log2_probability"] = {"numerator": -493, "denominator": 4}
        self.assertAlmostEqual(cost_model(model).latent_path_bits, 123.25)
        model["marginal_log2_probability"] = {"numerator": 1, "denominator": 1}
        with self.assertRaises(CodecError):
            cost_model(model)
        model["marginal_log2_probability"] = -123
        with self.assertRaises(CodecError):
            cost_model(model)

    def test_expected_conformance_vector(self):
        report = cost_model(self.model()).as_dict()
        expected = {
            "canonical_serialization_bits": 2577,
            "partition_bits": 8.169925001442314,
            "topology_bits": 13.0,
            "transition_kt_bits": 16.00783198447172,
            "emission_kt_bits": 29.78171197699629,
            "latent_path_bits": 19.0,
            "external_model_index_bits": 1.584962500721156,
            "structural_universal_bits": 87.5444314636315,
        }
        self.assertEqual(report["canonical_serialization_bits"], expected["canonical_serialization_bits"])
        for key, value in expected.items():
            if key != "canonical_serialization_bits":
                self.assertAlmostEqual(report[key], value, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
