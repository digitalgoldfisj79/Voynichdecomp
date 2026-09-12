#!/usr/bin/env python3
"""Independent decoder for Yiddish qualification v1b M0 controls.

Intentionally contains no encoder or solver import and implements inversion/decoding
from first principles for round-trip and oracle checks.
"""

def invert_permutation(plain_to_cipher):
    if sorted(plain_to_cipher) != list(range(len(plain_to_cipher))):
        raise ValueError('key is not a permutation')
    cipher_to_plain=[None]*len(plain_to_cipher)
    for plain,cipher in enumerate(plain_to_cipher):
        if cipher_to_plain[cipher] is not None:
            raise ValueError('non-bijective key')
        cipher_to_plain[cipher]=plain
    return cipher_to_plain


def decode_words(cipher_words, cipher_to_plain):
    out=[]
    for word in cipher_words:
        dec=[]
        for ch in word:
            if ch == '~':
                dec.append('~')
            elif 'a' <= ch <= 'z':
                dec.append(chr(97 + cipher_to_plain[ord(ch)-97]))
            else:
                raise ValueError(f'unregistered cipher atom {ch!r}')
        out.append(''.join(dec))
    return out


def assert_roundtrip_non_erased(truth_words, decoded_words):
    if len(truth_words) != len(decoded_words):
        raise AssertionError('word count mismatch')
    for truth,decoded in zip(truth_words,decoded_words):
        if len(truth) != len(decoded):
            raise AssertionError('atom count mismatch')
        for a,b in zip(truth,decoded):
            if b != '~' and a != b:
                raise AssertionError('roundtrip mismatch')
