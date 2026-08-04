# f116v transcription apparatus v0.1

## Status vocabulary

- `PROBABLE`: CATMuS assigned the same character at an aligned cut in true-colour and monochrome-PCA views, both confidences exceeded the blank-derived threshold, and local edge correlation passed.
- `<x?>`: cross-view shape/label agreement exists but one confidence or structural gate is weaker.
- `[x|y]`: both views support an acquired mark but assign different labels.
- `<?>`: unresolved position.
- `…`: unresolved span of indeterminate length.

No item in this document is `SUPPORTED` under the frozen protocol because the independent TrOCR architecture did not provide adequate positional corroboration.

## Line 1

Apparatus:

`<unresolved>`

CATMuS whole-line hypotheses:

- true colour: `poe sebet umor antgte`
- monochrome PCA: `ra bober umer mtt`
- colour PCA: `Iledetedaig ete`

Interpretation: no high-confidence, positionally stable character passed the corrected gate. The line should not be transcribed from these models.

## Line 2

Apparatus:

`… <n?> c h i c o <n?> [o|e] l a d a b a … s … r [c|d] <e?> r e … <p?> o <r?> …`

Probable character labels, in positional order:

`c h i c o l a d a b a s r e o`

High-confidence ambiguous positions:

- `o|e` immediately before the `ladaba` sequence;
- `c|d` in the later `…r[c|d]<e?>re…` sequence.

Lower-confidence but cross-view aligned positions include the `n` before `chico`, the `n` after it, `e`, `p`, and `r` elsewhere in the line.

CATMuS whole-line hypotheses:

- true colour: `E⁊ anchicon oladabas taualces ⁊ re tcer cerea ppor tad ⁊ago`
- monochrome PCA: `s inchicon eladabao t nlcos cecor deres porta`
- colour PCA: `lab`

Independent TrOCR hypotheses were unstable. Their local overlap is insufficient to choose between the alternatives or validate spaces.

Interpretation: the physical writing supports a central sequence compatible with `…chicon…ladaba…`, but not a complete reading. The model's `c` in `chicon` must not be silently changed to `t` merely to match a familiar human transcription.

## Line 3

Apparatus:

`… <t?> … a r i <x?> … <m?> o u <x?> <t?> <u?> [x|o] … <l?> …`

Probable labels:

`a r i o u`

CATMuS whole-line hypotheses:

- true colour: `Lsixtanarix ⁊ mouyxt uixt alar inauat`
- monochrome PCA: `itrarixł mouxt uopale dlo`
- colour PCA: `egt g permet A`

Interpretation: the recurrent-looking cores `ari` and `ou` are more stable than the surrounding labels, but their word boundaries and neighbouring letters are unresolved.

## Line 4

Apparatus:

`… <c?> <o?> <t?> … p a … <e?> a u … r e <n?> … <s?> o <combining-mark?> … <u?> <i?> g a …`

Probable labels:

`p a a u r e o g a`

CATMuS whole-line hypotheses:

- true colour: `Ma corta palbea ulren sõ nui gas michꝰ o`
- monochrome PCA: `ILao coto patea ubren ssõ mui galtt ou ł`
- colour PCA: `gdaindoise ii de n`

Interpretation: several local shapes recur at aligned positions, but the line-level strings diverge heavily. The sequences `pa`, `au`, `re`, `o`, and `ga` are provisional local labels, not validated words.

## Explicit non-findings

- No complete line is independently validated.
- No language identification follows from the model outputs.
- No word division is validated.
- No abbreviation is expanded.
- No erased text is incorporated.
- No material conclusion about retracing or multiple phases follows from this apparatus.
