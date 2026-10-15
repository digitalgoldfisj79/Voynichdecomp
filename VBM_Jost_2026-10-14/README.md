# VBM – Analyse und Dateien für Jost (14. Oktober 2026)

**Stand: abgeschlossen.** Dieses Verzeichnis enthält die **einzeln ausgepackten Dateien** aus dem VBM-Forschungspaket: 316 Original-, Analyse-, Programm-, Test- und Ergebnisdateien. Ein Binär-ZIP ist für den Zugriff nicht erforderlich.

## Inhalt

- [Alle Einzelfiles und Verzeichnisse](./Material/)
- [Originalgenerator v138](./Material/01_Original/Generator_v138_Pruefpaket_2026-10-14/gen138/)
- [Deutscher Paketindex](./Material/LESEN_ZUERST_DE.md)
- [E-Mail-Entwurf für Jost](./Material/E-Mail_an_Jost_DE.txt)
- [Direktaudit v138](./01_v138_Originalgenerator_Audit.md)
- [Falsifizierbarkeits- und Identifizierbarkeitsaudit](./02_VBM_Falsifizierbarkeit.md)
- [v142: Unabhängige Dekodierung](./03_v142_Decodierbarkeit.md)
- [v143: Wortübergreifender Zustand](./04_v143_Wortuebergreifender_Zustand.md)
- [v145: Nahtkompatible Codebücher](./05_v145_Nahtkompatible_Codebooks.md)
- [v146: Abkürzungen und Schreibbarkeit](./06_v146_Abkuerzungssysteme.md)
- [v147: Kapazitätsschranken](./07_v147_Kapazitaetsgrenzen.md)
- [Schreibpraktische Prüfung](./08_Schreibpraktische_Pruefung.md)
- [Manifest und SHA-256-Werte](./Material/MANIFEST_SHA256.csv)

## Gesamturteil

**Die konkrete v138-Implementierung erfüllt die geprüften mechanischen und statistischen Anforderungen nicht.** Der Generator ist reproduzierbar, zeigt aber Kollisionszellen, nicht konsistente Zustandsfortschreibung und nicht zugelassene Zeichenfolgen. Die AUC-Schwelle akzeptierte in allen 16 Versuchen auch innerhalb der Zeilen vertauschte echte Voynich-Wörter. Ein direkter Test der originalen Brückenannahme ergab keinen belastbaren positiven Befund.

**Die zugrunde liegende allgemeine VBM-Hypothese ist dadurch nicht endgültig widerlegt.** Die späteren reversiblen Versuchssysteme implementieren teilweise andere Brücken- und Wortgrenzenregeln und identifizieren den historischen Mechanismus nicht. Die Arbeit an weiteren Generatorversionen wurde deshalb eingestellt.

## Sonderfall: große JSON-Datei

**Alle 316 Quelldateien sind im Branch vertreten.** Lediglich die 4.483.164 Byte große `voynich_transcriptions_slim.json` liegt platzsparend als [`voynich_transcriptions_slim.json.gz`](./Material/05_v145_bis_v147/VBM_v145_Seam_Compatible_Codebooks_2026-10-14/vbm_v145/inputs/voynich_transcriptions_slim.json.gz) vor, zusammen mit [`RESTORE_DATA.txt`](./Material/05_v145_bis_v147/VBM_v145_Seam_Compatible_Codebooks_2026-10-14/vbm_v145/inputs/RESTORE_DATA.txt). Zur Wiederherstellung: `gzip -dk voynich_transcriptions_slim.json.gz`. Nach dem Entpacken entspricht die Datei dem ursprünglichen SHA-256 `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`.

## Hinweis

Dieses GitHub-Repository ist **öffentlich**. Mit diesem Branch sind die enthaltenen Originaldateien des Autors ebenfalls öffentlich lesbar. Das Einverständnis zur Weiterverbreitung sollte unabhängig geprüft werden.
