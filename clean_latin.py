"""Build a properly cleaned Latin word list from the antidotarium."""
import re, json
from collections import Counter

with open('/mnt/user-data/uploads/antidotarium_latin.txt') as f:
    text = f.read()

all_words = re.findall(r'[a-zA-Z]+', text.lower())
print("Raw words: {}".format(len(all_words)))

# === FILTER 1: Remove any word containing 'w' (not a Latin letter) ===
# === FILTER 2: Remove any word containing 'ij' (Dutch) ===
# === FILTER 3: Remove single-letter words entirely ===
# === FILTER 4: 2-letter words only if known Latin ===

LATIN_2LETTER = frozenset([
    'et', 'in', 'ad', 'an', 'de', 'si', 'ex', 'ut', 'os', 'ur',
    'am', 'um', 'is', 'us', 'ae', 'at', 'ar', 'er', 'or', 'ab',
    'id', 'se', 'ac', 'ea', 'ei', 'eo', 'ne', 'te', 'me',
])

# === FILTER 5: Comprehensive Dutch/editorial/fragment blacklist ===
BLACKLIST = frozenset([
    # Dutch function words
    'ende','van','den','die','dat','men','het','dats','elcs','elx',
    'saelt','geven','wine','comt','nemt','oec','iegen','coude','dage','sal','sijn',
    'goet','hoefde','werso','ment','vint','sede','maect','alle','pine','lichame',
    'borst','deel','vandor','vander','dese','welke','eene','ghij','ghi','soe','daer',
    'hier','mede','boven','onder','voor','achter','binnen','buiten','sulc','mach',
    'doet','wert','hare','sine','zijne','zeem','rosen','olie','olye','doen','knde',
    'riet','saet','wast','noch','hem','ons','als','alse','eer','hoe','wie','wat',
    'waer','ene','dien','dier','mij','suldi','nde','gnouch','datt','tsop',
    'deser','niet','meer','ook','heeft','wordt','deze','alleen','eerst','volgt',
    'plaats','daarvan','verder','staat','texten','hebben','andere','hiervan',
    'breekt','staan','korter','ongeveer','hetzelfde','verkort','ontbreekt',
    'behoort','overeen','inhoud','volgens','vergel','opvallend','duidelijk',
    'schijnlijk','behoorende','overeenkomt','bij','nog','toe','tot','uit',
    'dan','wel','naar','zijn','maar','dus','toch','toen','want','omdat',
    'zonder','tussen','tegen','sinds','langs','over','onder','naar',
    # Editorial abbreviations (manuscript sigla)
    'fj','fi','nm','mt','lj','lt','ft','ru','fol',
    # Roman numerals
    'ii','iii','iv','vi','vii','viii','ix','xi','xii','xiii','xiv','xv',
    'xvi','xvii','xviii','xix','xx','iiii','viiii','xl',
    # Editorial notation
    'ai','gr','re','es','vc','iq','ke','kcr','rij','iij',
    # French fragments (some editorial notes in French)
    'les','des','est','une','pour','avec','soit','fait','elle',
    'qui','ont','sont','fut','puis','eet',
    # More Dutch
    'nog','ook','bij','tot','dan','wel','naar','hun','ons','hem',
    'haar','dit','wat','hoe','wie','maar','dus','toch','toen',
    'reeds','steeds','slechts','zooals','evenaals','echter',
    'dese','suldi','iegew','doete','deser','gesodew','prumew',
    'gemiwct','fleumew','polipodiuw','dorgatew','bereidingswijze',
    'gewichtsteeken','overbodig','corrupte','opvallend','ingevuld',
    'zondernaam','nergens','geplaatst','aanbevolen','aangeraden',
    'gecompli','simplicia','vergelijking','overeenkomstige',
    'middelnederl','alleen','toeg','daarvan','geheele',
    'verkort','vernemen',
])

# === FILTER 6: Remove words that look like Dutch (heuristic patterns) ===
def looks_dutch(w):
    if 'w' in w: return True
    if 'ij' in w: return True
    if 'oo' in w and w not in ('coctionem','cooperiantur','cooperto'): return True
    if 'aa' in w and w not in ('balaustie','balaustias','aaron'): return True
    if w.endswith('ght'): return True
    if w.endswith('sch') and w not in ('muscsch'): return True
    if w.endswith('ck'): return True
    if w.endswith('ew'): return True
    return False

kept = []
removed_reasons = Counter()
for w in all_words:
    if len(w) < 2:
        removed_reasons['too_short'] += 1
        continue
    if len(w) == 2 and w not in LATIN_2LETTER:
        removed_reasons['non_latin_2char'] += 1
        continue
    if w in BLACKLIST:
        removed_reasons['blacklist'] += 1
        continue
    if looks_dutch(w):
        removed_reasons['dutch_pattern'] += 1
        continue
    kept.append(w)

print("After cleaning: {} words".format(len(kept)))
print("Removed reasons:")
for reason, count in removed_reasons.most_common():
    print("  {}: {}".format(reason, count))

# Split into recipes
recipes = []
current = []
for w in kept:
    if w == 'recipe' and current:
        recipes.append(current)
        current = []
    current.append(w)
if current:
    recipes.append(current)

all_clean = [w for r in recipes for w in r]
freq = Counter(all_clean)
print("\nCleaned: {} words, {} types, {} recipes".format(
    len(all_clean), len(freq), len(recipes)))

# Quick check: top 30 words
print("\nTop 30 words:")
for w, c in freq.most_common(30):
    print("  {}: {}".format(w, c))

# Save
with open('/home/claude/clean_latin_recipes.json', 'w') as f:
    json.dump({'recipes': recipes, 'stats': {
        'n_words': len(all_clean), 'n_types': len(freq), 'n_recipes': len(recipes)
    }}, f)
print("\nSaved: clean_latin_recipes.json")
