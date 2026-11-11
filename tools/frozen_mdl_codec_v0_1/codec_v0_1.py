#!/usr/bin/env python3
"""Frozen MDL codec v0.1 reference implementation.

Canonical prefix-free value serialization, enumerative structure costs,
Jeffreys/Krichevsky–Trofimov categorical costs, and explicit latent-path
charging. This module reads no Voynich data.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

VERSION = "frozen-mdl-codec-v0.1"
T_NULL, T_FALSE, T_TRUE, T_INT, T_STR, T_LIST, T_DICT = range(7)


class CodecError(ValueError):
    pass


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def bit(self, value: int) -> None:
        if value not in (0, 1):
            raise CodecError("bit must be 0 or 1")
        self.bits.append(value)

    def uint(self, value: int, width: int) -> None:
        if value < 0 or value >= (1 << width):
            raise CodecError("integer does not fit width")
        for shift in range(width - 1, -1, -1):
            self.bit((value >> shift) & 1)

    def raw(self, payload: bytes) -> None:
        for byte in payload:
            self.uint(byte, 8)

    def finish(self) -> tuple[bytes, int]:
        length = len(self.bits)
        bits = self.bits + [0] * ((8 - length % 8) % 8)
        output = bytearray()
        for start in range(0, len(bits), 8):
            value = 0
            for bit in bits[start:start + 8]:
                value = (value << 1) | bit
            output.append(value)
        return bytes(output), length


class BitReader:
    def __init__(self, payload: bytes, bit_length: int) -> None:
        if bit_length < 0 or bit_length > 8 * len(payload):
            raise CodecError("invalid bit length")
        self.bits = [
            (payload[index // 8] >> (7 - index % 8)) & 1
            for index in range(bit_length)
        ]
        self.offset = 0

    def bit(self) -> int:
        if self.offset >= len(self.bits):
            raise CodecError("unexpected end of bit stream")
        value = self.bits[self.offset]
        self.offset += 1
        return value

    def uint(self, width: int) -> int:
        value = 0
        for _ in range(width):
            value = (value << 1) | self.bit()
        return value

    def raw(self, length: int) -> bytes:
        return bytes(self.uint(8) for _ in range(length))

    @property
    def exhausted(self) -> bool:
        return self.offset == len(self.bits)


def write_gamma(writer: BitWriter, n: int) -> None:
    if n < 1:
        raise CodecError("Elias gamma requires n >= 1")
    width = n.bit_length()
    for _ in range(width - 1):
        writer.bit(0)
    writer.uint(n, width)


def read_gamma(reader: BitReader) -> int:
    zeros = 0
    while reader.bit() == 0:
        zeros += 1
    suffix = reader.uint(zeros) if zeros else 0
    return (1 << zeros) | suffix


def write_elias_delta(writer: BitWriter, n: int) -> None:
    if n < 1:
        raise CodecError("Elias delta requires n >= 1")
    width = n.bit_length()
    write_gamma(writer, width)
    if width > 1:
        writer.uint(n ^ (1 << (width - 1)), width - 1)


def read_elias_delta(reader: BitReader) -> int:
    width = read_gamma(reader)
    suffix = reader.uint(width - 1) if width > 1 else 0
    return (1 << (width - 1)) | suffix


def write_nat(writer: BitWriter, n: int) -> None:
    if n < 0:
        raise CodecError("natural number must be non-negative")
    write_elias_delta(writer, n + 1)


def read_nat(reader: BitReader) -> int:
    return read_elias_delta(reader) - 1


def zigzag_encode(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1


def zigzag_decode(n: int) -> int:
    return n // 2 if n % 2 == 0 else -(n // 2) - 1


def validate(value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, list):
        for item in value:
            validate(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CodecError("dictionary keys must be strings")
        for item in value.values():
            validate(item)
        return
    raise CodecError(f"unsupported value type: {type(value).__name__}")


def encode(writer: BitWriter, value: Any) -> None:
    validate(value)
    if value is None:
        writer.uint(T_NULL, 3)
    elif value is False:
        writer.uint(T_FALSE, 3)
    elif value is True:
        writer.uint(T_TRUE, 3)
    elif isinstance(value, int):
        writer.uint(T_INT, 3)
        write_nat(writer, zigzag_encode(value))
    elif isinstance(value, str):
        writer.uint(T_STR, 3)
        payload = value.encode("utf-8")
        write_nat(writer, len(payload))
        writer.raw(payload)
    elif isinstance(value, list):
        writer.uint(T_LIST, 3)
        write_nat(writer, len(value))
        for item in value:
            encode(writer, item)
    elif isinstance(value, dict):
        writer.uint(T_DICT, 3)
        items = sorted(value.items(), key=lambda pair: pair[0].encode("utf-8"))
        write_nat(writer, len(items))
        for key, item in items:
            encode(writer, key)
            encode(writer, item)
    else:  # pragma: no cover
        raise CodecError("unsupported value")


def decode(reader: BitReader) -> Any:
    tag = reader.uint(3)
    if tag == T_NULL:
        return None
    if tag == T_FALSE:
        return False
    if tag == T_TRUE:
        return True
    if tag == T_INT:
        return zigzag_decode(read_nat(reader))
    if tag == T_STR:
        try:
            return reader.raw(read_nat(reader)).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CodecError("invalid UTF-8") from exc
    if tag == T_LIST:
        return [decode(reader) for _ in range(read_nat(reader))]
    if tag == T_DICT:
        result: dict[str, Any] = {}
        previous: bytes | None = None
        for _ in range(read_nat(reader)):
            key = decode(reader)
            if not isinstance(key, str):
                raise CodecError("dictionary key is not a string")
            encoded_key = key.encode("utf-8")
            if previous is not None and encoded_key <= previous:
                raise CodecError("non-canonical dictionary order")
            result[key] = decode(reader)
            previous = encoded_key
        return result
    raise CodecError("reserved type tag")


def serialize(value: Any) -> tuple[bytes, int]:
    writer = BitWriter()
    encode(writer, value)
    return writer.finish()


def deserialize(payload: bytes, bit_length: int) -> Any:
    reader = BitReader(payload, bit_length)
    value = decode(reader)
    if not reader.exhausted:
        raise CodecError("trailing bits")
    return value


def serialized_bit_length(value: Any) -> int:
    return serialize(value)[1]


def elias_delta_length(n: int) -> int:
    writer = BitWriter()
    write_elias_delta(writer, n)
    return len(writer.bits)


def nat_length(n: int) -> int:
    return elias_delta_length(n + 1)


def log2_binomial(n: int, k: int) -> float:
    if n < 0 or k < 0 or k > n:
        raise CodecError("invalid binomial arguments")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def log2_multinomial(counts: Sequence[int]) -> float:
    if any(count < 0 for count in counts):
        raise CodecError("negative count")
    return (math.lgamma(sum(counts) + 1) - sum(math.lgamma(c + 1) for c in counts)) / math.log(2)


def partition_codelength(class_sizes: Sequence[int]) -> float:
    if not class_sizes or any(size <= 0 for size in class_sizes):
        raise CodecError("class sizes must be positive")
    classes, total = len(class_sizes), sum(class_sizes)
    return nat_length(classes) + log2_binomial(total - 1, classes - 1) + log2_multinomial(class_sizes)


def state_topology_codelength(num_states: int, outdegrees: Sequence[int]) -> float:
    if num_states <= 0 or len(outdegrees) != num_states:
        raise CodecError("one outdegree is required per state")
    total = float(nat_length(num_states))
    for degree in outdegrees:
        if degree < 0 or degree > num_states:
            raise CodecError("invalid outdegree")
        total += nat_length(degree) + log2_binomial(num_states, degree)
    return total


def kt_codelength(counts: Sequence[int], alpha: float = 0.5) -> float:
    if not counts or alpha <= 0 or any(count < 0 for count in counts):
        raise CodecError("invalid categorical counts")
    k, total = len(counts), sum(counts)
    log_probability = (
        math.lgamma(k * alpha)
        - math.lgamma(total + k * alpha)
        + sum(math.lgamma(count + alpha) - math.lgamma(alpha) for count in counts)
    )
    return -log_probability / math.log(2)


def rowwise_kt_codelength(rows: Sequence[Sequence[int]]) -> float:
    return sum(kt_codelength(row) for row in rows)


def explicit_latent_path_codelength(path: Sequence[int], num_states: int, reset_points: Iterable[int] = ()) -> float:
    if num_states <= 0 or any(state < 0 or state >= num_states for state in path):
        raise CodecError("invalid state alphabet or path")
    resets = sorted(set(reset_points))
    if any(point < 0 or point >= len(path) for point in resets):
        raise CodecError("invalid reset point")
    if not path:
        return nat_length(num_states) + nat_length(0)
    reset_set = set(resets)
    rows = [[0] * num_states for _ in range(num_states)]
    uniform_events = 1
    for index in range(1, len(path)):
        if index in reset_set:
            uniform_events += 1
        else:
            rows[path[index - 1]][path[index]] += 1
    return (
        nat_length(num_states)
        + nat_length(len(resets))
        + sum(nat_length(point) for point in resets)
        + uniform_events * math.log2(num_states)
        + rowwise_kt_codelength(rows)
    )


def exact_rational(record: Any, field: str) -> float:
    if not isinstance(record, dict) or set(record) != {"numerator", "denominator"}:
        raise CodecError(f"{field} must be an exact rational record")
    numerator, denominator = record["numerator"], record["denominator"]
    if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
        raise CodecError(f"invalid exact rational in {field}")
    return numerator / denominator


@dataclass(frozen=True)
class CostReport:
    canonical_serialization_bits: int
    partition_bits: float
    topology_bits: float
    transition_kt_bits: float
    emission_kt_bits: float
    latent_path_bits: float
    external_model_index_bits: float

    @property
    def structural_universal_bits(self) -> float:
        return sum((self.partition_bits, self.topology_bits, self.transition_kt_bits, self.emission_kt_bits, self.latent_path_bits, self.external_model_index_bits))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "canonical_serialization_bits": self.canonical_serialization_bits,
            "partition_bits": self.partition_bits,
            "topology_bits": self.topology_bits,
            "transition_kt_bits": self.transition_kt_bits,
            "emission_kt_bits": self.emission_kt_bits,
            "latent_path_bits": self.latent_path_bits,
            "external_model_index_bits": self.external_model_index_bits,
            "structural_universal_bits": self.structural_universal_bits,
        }


def cost_model(model: Mapping[str, Any]) -> CostReport:
    if model.get("codec_version") != VERSION:
        raise CodecError(f"codec_version must be {VERSION!r}")
    class_sizes = model.get("class_sizes", [])
    num_states = int(model.get("num_states", 0))
    outdegrees = model.get("outdegrees", [])
    transition_rows = model.get("transition_counts", [])
    emission_rows = model.get("emission_counts", [])

    latent_mode = model.get("latent_path_mode", "none")
    if latent_mode == "none":
        latent_bits = 0.0
    elif latent_mode == "explicit":
        latent_bits = explicit_latent_path_codelength(model.get("latent_path", []), num_states, model.get("reset_points", []))
    elif latent_mode == "marginalized":
        value = exact_rational(model.get("marginal_log2_probability"), "marginal_log2_probability")
        if value > 0:
            raise CodecError("marginal log2 probability must be <= 0")
        latent_bits = -value
    else:
        raise CodecError("invalid latent_path_mode")

    model_count = int(model.get("external_model_count", 1))
    model_index = int(model.get("external_model_index", 0))
    if model_count <= 0 or model_index < 0 or model_index >= model_count:
        raise CodecError("invalid external model index")

    return CostReport(
        canonical_serialization_bits=serialized_bit_length(dict(model)),
        partition_bits=partition_codelength(class_sizes) if class_sizes else 0.0,
        topology_bits=state_topology_codelength(num_states, outdegrees) if num_states else 0.0,
        transition_kt_bits=rowwise_kt_codelength(transition_rows) if transition_rows else 0.0,
        emission_kt_bits=rowwise_kt_codelength(emission_rows) if emission_rows else 0.0,
        latent_path_bits=latent_bits,
        external_model_index_bits=math.log2(model_count),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    payload, bit_length = serialize(model)
    if args.binary:
        args.binary.write_bytes(payload)
    if args.meta:
        args.meta.write_text(json.dumps({"codec_version": VERSION, "bit_length": bit_length}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cost_model(model).as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
