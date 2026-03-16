# HA Folio-by-Folio CV Enrichment Analysis
## Date: 2026-03-15

### Method
For each of 48 HA folios, test every consonant+vowel (CV) pair for 
hypergeometric enrichment against the HA-wide baseline. Bonferroni 
threshold: p < 0.0010 (0.05/48).

### Bonferroni-significant hits (p < 0.001)

| Folio | CV | ×n | Folio rate | HA rate | Enrichment | p-value | Candidate Latin |
|-------|----|----|-----------|---------|------------|---------|----------------|
| **f2r** | **mi** | 8 | 17.4% | 1.7% | **10.4×** | 3.7e-07 | minoris/minor/misce |
| **f8v** | bo | 2 | 3.8% | 0.1% | 33.4× | 8.8e-04 | ? (b+o) |
| **f17v** | **fe** | 4 | 6.2% | 0.6% | **10.7×** | 3.2e-04 | feniculi/fel/febribus |
| **f20r** | vi | 3 | 6.2% | 0.4% | 15.5× | 6.4e-04 | ? (v+i) |
| **f23r** | **te** | 3 | 8.3% | 0.5% | **18.1×** | 4.3e-04 | terantur/tempore |
| **f23v** | **te** | 3 | 7.7% | 0.5% | **16.7×** | 5.4e-04 | terantur/tempore |

### Interpretation

**f2r: CONFIRMED — Centaurea minor** (from previous session, p<10⁻⁶)
'mi' = minoris/minor. The illustration matches. Cross-validated independently.

**f17v: fe = feniculi/fel/febribus**
Previously tested as fennel — FALSIFIED on illustration (illustration = Tamus communis,
not Foeniculum). But 'fe' enrichment is real. Either:
(a) the text discusses fennel despite the illustration being Tamus, or
(b) 'fe' encodes a different f+e word (febribus = fevers?)
Note: f17v also shows de×7 (p=0.002) = decoctio/decoctum. A recipe page about
decoctions for fevers? Consistent with pharmaceutical Latin.

**f23r + f23v: te = terantur/tempore**
Same bifolio, same signal. 'te' = t+e words. In CI, "terantur" (let them be ground)
is the most common t+e word. A preparation page about grinding/processing.
f23v also shows tu×3 (p=0.005) = tundas/tussim (pounding/cough).
f23r+f23v together: grinding, pounding, time-related preparation instructions.

**f8v: bo (unassigned)**
b+o is very rare in HA (0.1% baseline). Row 'r' contains {b,z,x,j,k,w,y}.
Possible: botanical/borago/bolus? Too few counts (n=2) to be confident.

**f20r: vi (unassigned)**
v+i words. Row 'c' contains {∅,v}. Possible: vinum/vino/virtus/viridis?

### Notable sub-Bonferroni patterns (p < 0.01)

| Folio | CV | ×n | p-value | Candidate |
|-------|----|----|---------|-----------|
| f1r | su | 6 | 0.004 | succum/succo (juice) |
| f1r | ca | 10 | 0.009 | calida/caput/carnem (hot/head/flesh) |
| f3r | do | 5 | 0.007 | ? (d+o) |
| f9r | ci | 5 | 0.004 | cibis/clister (food/enema) |
| f10v | ci | 4 | 0.002 | cibis/clister |
| f13v | su | 3 | 0.006 | succum/succo |
| f17v | de | 7 | 0.002 | decoctio/decoctum |
| f19r | ∅e | 7 | 0.002 | educit/eius/efficacia |
| f19v | ∅u | 4 | 0.004 | uino/ueteri/uim (wine/old/strength) |
| f20v | ci | 5 | 0.007 | cibis/clister |
| f20v | pa | 4 | 0.009 | pars/panno/patienti |
| f21r | do | 4 | 0.006 | ? |
| f21v | cu | 8 | 0.006 | cutem/cura/curatur (skin/cure) |
| f21v | sa | 5 | 0.007 | sal/sanat/sanguinem (salt/heals/blood) |
| f23v | tu | 3 | 0.005 | tundas/tussim (pound/cough) |
| f24r | fu | 5 | 0.002 | fuerunt/frutex (shrub) |
| f24v | ru | 4 | 0.003 | ? (r+u) |
| f25r | ∅i | 3 | 0.006 | ipso/ibi/ipsius |

### Thematic clusters

**Preparation vocabulary:** f17v(de=decoction, fe=fennel?), f23r+f23v(te=grinding, tu=pounding)
**Juice/liquid:** f1r(su=succum), f13v(su=succum), f19v(∅u=vino)
**Medical conditions:** f21v(cu=cutem/skin, sa=sanguinem/blood)
**Food/diet:** f9r+f10v+f20v(ci=cibis/food)
**Oil/fat:** f11r+f13r+f18r+f22r+f23r(∅o=oleo/oleum — appears on 5 folios)

### What this means for decipherment
These are NOT readings. They are consonant-group + vowel constraints.
But several folios show clear pharmaceutical Latin topic signatures
that are consistent with Circa Instans-like content.

The f23r/f23v bifolio is the strongest new lead — same CV enriched on
both sides of the same leaf, encoding preparation instructions about
grinding/pounding. That's exactly what a pharmaceutical recipe would contain.
