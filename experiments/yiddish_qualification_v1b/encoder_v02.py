#!/usr/bin/env python3
"""Encoder for Yiddish qualification v1b M0 controls. No solver imports."""

def encode_words(words, plain_to_cipher, erasure_rate, rng):
    out=[]; erased=0; total=0
    for word in words:
        enc=[]
        for ch in word:
            p=ord(ch)-97
            total += 1
            if erasure_rate and rng.random() < erasure_rate:
                enc.append('~'); erased += 1
            else:
                enc.append(chr(97 + plain_to_cipher[p]))
        out.append(''.join(enc))
    return out, erased, total
