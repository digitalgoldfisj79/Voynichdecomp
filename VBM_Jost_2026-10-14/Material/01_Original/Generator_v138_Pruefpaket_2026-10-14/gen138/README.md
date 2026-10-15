# Generator v138 - Pruefpaket fuer Gegentest (Stand 2026-10-14)

Zweck: Ein unabhaengiger Pruefer soll (a) den Generator byteidentisch neu laufen lassen,
(b) die Kennzahlen gegen das VMS nachrechnen und (c) die Eigenschaften pruefen, die wir
behaupten: deterministisch, rueckwaerts lesbar bis auf gezaehlte Verluste, nur im VMS
belegte Zeichenfolgen, Abnahmemass unter der Schwelle.

## Was v138 ist (in Worten)

v138 verwandelt einen deutschen (bairischen, 15. Jh.) Text in einen Text aus Voynich-Zeichen,
mit vier festen Tafeln und ohne Zufall. Lesung des VMS ist REVA (Zeile und Token gespiegelt;
EVA qo wird o,q). Einheit des Klartexts ist das Ereignis Vokal - Konsonantencluster - Vokal.

1. Clustertafel (Blatt 1). Alle Konsonantencluster des Klartexts werden nach Haeufigkeit
   geordnet und nach Rang auf die 443 Zellen (Schluessel, Zustand, Kopf) des VMS-B gelegt,
   die ebenfalls nach Haeufigkeit geordnet sind: haeufige Cluster auf haeufige Zellen, seltene
   auf seltene. Jeder Cluster hat genau eine Zelle. Cluster jenseits der 443 werden zyklisch
   auf seltene Zellen gelegt. Die Zuordnung ist Rang, nicht Deutung.
2. Vokaltafel (Blatt 2). Der Vokal zwischen zwei Ereignissen wird als Paar geschrieben:
   Bruecke am Ende des Tokens, Schluessel am Anfang des naechsten. Jedes Paar steht fuer genau
   einen Vokal (belegung.json, Fit vom 2026-10-10, ergaenzt um die beweglichen Schliesser
   lo/la/ro/ra). Welches Paar ein Vokal bekommt, entscheiden Kopfklasse der laufenden Zelle
   und Familie der naechsten Zelle (y / n / r-l), nach VMS-Haeufigkeit; das Paar muss einen
   Schluessel haben, den die naechste Zelle traegt.
3. Wortfugen (Vorbau-Pool, Blatt 3). Enthaelt der Cluster eine Wortgrenze des Klartexts und
   hat er drei oder mehr Konsonanten, schreibt der Generator die Zelle des Auslauts und haengt
   den Anlaut des naechsten Worts als Vorbau an. Vorbau-Formen sind ganze VMS-Vorbaue, je
   Zelle (Zustand, Kopf) nach Haeufigkeit sortiert; der Anlaut waehlt nach Rang. Reichen die
   Formen nicht, bekommt der Anlaut die Form seiner Klasse (erstes Zeichen) - das ist der
   gezaehlte Verlust (v138: 1791 von 13653 Fugen). Lange Binnencluster bekommen ihren Rest als
   Vorbau nach Rang (rest_stuecke_dt.json -> vorbau_strings_vms.json), erst ab zwei Zeichen.
4. Nahtregel. Jedes Token darf nur Atom-Zweier- und Dreierfolgen enthalten, die im VMS-B
   wenigstens einmal stehen. Verstoesst eine Zusammensetzung, nimmt der Generator die naechste
   Vorbau-Form desselben Pools, dann das naechste Paar desselben Vokals, sonst laesst er den
   Vorbau weg (gezaehlt: 4468 / 565 / 116).
5. Weitere feste Regeln: leere Cluster (Vokal auf Vokal) werden als blosses Paar geschrieben
   (kopfloses Token); bewegliche Schliesser lo/la/ro/ra werden nach den VMS-Space-Raten als
   eigenes Token abgetrennt (deterministisch ueber einen Laufzaehler); "vnd" wird als s-Token
   geschrieben (Crib); Zeilenumbruch nach VMS-Zeilenlaengen (gen.wrap); am EVA-Zeilenanfang
   eines Absatzes eine Marke p/t/k/f (REVA: letztes Atom der Zeile) mit festem Zufallssaat 1420
   fuer Absatzlaenge und Markenwahl - das ist die einzige Stelle mit random, und sie ist gesaet.

v140 ist v138 plus Homophone: jeder Cluster bekommt nach seiner Masse weitere Zellen, der
Schreiber nimmt reihum die naechste (HOMK = Anteil zusaetzlicher Zellen, 0 = v138).

## Dateien

- vbm_tafel_v138.py, vbm_tafel_v140.py - die Generatoren
- Paket_v100/vbm_tafel_v119det.py - Basismodul (Blatt-1-Bau, Zerlegung, Zeilenumbruch); v138 importiert es
- panel_v126.py - Zerlegung der Token (cellparse) und Panel-Masse
- belegung.json (Paar -> Vokal), vorbau_strings_vms.json, rest_stuecke_dt.json
- vms_B_REVA_atomstream.txt - Currier B des VMS als REVA-Zeilen (2557 Zeilen, 21679 Token);
  daten/vms_B_odd.txt, vms_B_even.txt - Haelften fuer die Deckung
- klartext/bairisch_kurzertext_GROSS.txt - der Klartext; klartext/hartlieb_klartext.txt -
  Hartlieb-Kraeuterbuch (Lesetranskription mit Fehlern) fuer v140
- Paket_v100/mess_v118.py - Abnahme (h2-Blockluecke, AUC, Typen 21522, Hapax, Deckung)
- GENERAT_v138_kurzertext_REFERENZ.txt - das Generat, wie wir es erzeugt haben
- PANEL_v138.txt - unser Panel-Ausdruck

## Laufen lassen (Python 3.10+, numpy, scipy, scikit-learn fuer mess_v118)

    python3 vbm_tafel_v138.py                      # schreibt GENERAT_v138_kurzertext.txt
    cmp GENERAT_v138_kurzertext.txt GENERAT_v138_kurzertext_REFERENZ.txt   # muss leer sein
    python3 Paket_v100/mess_v118.py vms_B_REVA_atomstream.txt GENERAT_v138_kurzertext.txt
    python3 panel_v126.py vms_B_REVA_atomstream.txt GENERAT_v138_kurzertext.txt
    HOMK=0.1 python3 vbm_tafel_v140.py             # Hartlieb mit schwacher Homophonie

Erwartete Werte v138: h2 2,062, Luecke +0,094, AUC 0,704; Typen 3084; Hapax 51,4;
Deckung 64,2 / 64,4; Vorbau 26,7 %; kopflos 7,0 %; Paar-Restentropie 1,08 bit.
VMS-B zum Vergleich: Typen 4550, Hapax 68, Vorbau 25,1, kopflos 11,9, Paar 1,31.

## Was wir behaupten und was geprueft werden soll

1. Deterministisch: zweimal laufen = byteidentisch. Einzige Zufallsquelle ist die gesaete
   Markenschicht (Zeile ab "rng=random.Random(1420)").
2. Rueckwaerts lesbar: Blatt 1 ist eine Bijektion Cluster -> Zelle (bis auf die zyklisch
   belegten seltenen Zellen jenseits von 443 Clustern); Blatt 2 ordnet jedem Paar genau einen
   Vokal zu; der Vorbau ist je (Zelle, Anlaut) eindeutig bis auf die gezaehlten 1791
   Klassen-Rueckstufungen. Bitte pruefen, ob diese Aussagen im Code stimmen und ob es weitere
   Mehrdeutigkeiten gibt (Fremdpruefung vom 13.10. fand bei v135 drei; v136-v138 beheben sie).
3. Nur belegte Folgen: jedes Generat-Token enthaelt nur Atom-Bigramme und -Trigramme, die in
   vms_B_REVA_atomstream.txt vorkommen. Bitte gegenzaehlen.
4. Abnahme: AUC < 0,85 heisst, der h2-Blockverlauf des Generats ist vom VMS-B nicht sicher zu
   trennen. v138 liegt bei 0,704.

## Bekannte Schwaechen (nicht verschweigen)

- Typen 3084 gegen 4550 und Hapax 51 gegen 68: ein Viertel der VMS-Formenvielfalt fehlt.
- Fugen oq->y 94 % und a->r 89 % gegen VMS 77 / 48: Folge von genau einer Zelle je Cluster.
- Deckung (Wortformen, die auch im VMS stehen) 64 % ist teilweise Anpassung, weil der
  Vorbau-Pool aus VMS-Formen besteht; sie ist kein unabhaengiger Test.
- Blatt 1 und der Vorbau-Pool werden aus demselben VMS-B gebaut, gegen das gemessen wird.
  Kennzahlen, die direkt an Zelleninventar und Vorbau-Inventar haengen, sind deshalb nicht
  frei. Frei sind: h2-Verlauf, Typen, Hapax, Wiederholung von Tokenfolgen, Fugenmengen.
- Mit Fachtexten (Arzneibuecher, Hartlieb) als Klartext wird h2 besser und die Typenzahl
  schlechter; die Formelwiederholung eines Kraeuterbuchs schlaegt sichtbar durch
  (Dreierfolgen 10- bis 20-fach gegen VMS Herbal); v140 mit Homophonen unterdrueckt sie,
  flacht aber den Kopf der Verteilung ab (Top-20-Masse 16 % gegen 28 % in Herbal A).
- Der Klartext ist Fliessprosa, kein Herbar; die Hartlieb-Fassung enthaelt Lesefehler.

Die Frage, die v138 beantworten soll, ist Machbarkeit: kann ein deterministisches Tafelsystem,
das ein Schreiber des 15. Jahrhunderts fuehren koennte, aus deutschem Text einen Text machen,
der dem VMS-B in den Oberflaechenmassen nahe kommt und zurueckuebersetzbar bleibt. Es soll
nicht zeigen, dass das VMS so entstanden ist.
