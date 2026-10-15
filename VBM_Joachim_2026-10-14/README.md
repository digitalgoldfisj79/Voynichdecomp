# VBM – Auswertung für Joachim (14. Oktober 2026)

Diese Arbeitszweig-Dokumentation unterscheidet zwischen der **direkten Prüfung des originalen v138-Generators** und späteren, technisch anders konstruierten Versuchsmodellen.

## Kernaussage

**Original-v138 ist als geprüfte Implementierung nicht erfolgreich validiert.** Die Wiederholung seiner Ausgabe gelingt, aber die behauptete eindeutige Cluster-Zuordnung, konsistente Zustandsfortschreibung und durchgehend gültigen lokalen Glyphenfolgen sind im gelieferten Code verletzt. Der Ähnlichkeits-Score lässt umsortierten echten Voynich-Text als Negativkontrolle passieren. Der separate ursprüngliche Brücken-Test liefert keinen belastbaren positiven Befund.

**Diese negativen Befunde widerlegen nicht jede denkbare VBM mit Vokalbrücken über EVA-Leerzeichen.** Eine historische Entzifferung wurde nicht erreicht.

## Was spätere Varianten zeigen

- v142: Eigene, *innerhalb der Wörter* kodierende Varianten können 68.396 Entwicklungswörter und 13.232 Hartlieb-Wörter exakt dekodieren. Diese Varianten ersetzen wichtige Bausteine von Joachims Modell.
- v143/v144: Wortübergreifende Zustandsabhängigkeit ist synthetisch umsetzbar und nachweisbar, aber eine autonome Erzeugung kann dieselbe Ausgabe liefern.
- Prüfung des realen Manuskripts: Keine hinreichend robuste, abschnittsübergreifende Bestätigung des konkreten Verschlüsselungsmechanismus.
- v145: Weniger unzulässige lokale Glyphenfolgen, aber erzeugte Wörter durchschnittlich etwa acht statt fünf Zeichen lang.
- v146: Kürzere Wörter mit kompakten Tabellen, aber erhebliche Abweichungen in der Wortgrammatik.
- v147: Bestimmte feste Substitutionsmodelle sind unter der geprüften harten Trigramm-Bedingung ausgeschlossen; keine allgemeine Aussage gegen kompakte zustandsabhängige Verfahren.
- Scribal feasibility: Eine zeitgenössisch ausführbare Schreibprozedur, die **gleichzeitig** Reversibilität, Wortform, Länge, Gelenkstellen und überschaubare Tabellen erfüllt, ist nicht belegt.

## Dokumente auf diesem Branch

- [01_v138_Originalgenerator_Audit.md](./01_v138_Originalgenerator_Audit.md)
- [02_VBM_Falsifizierbarkeit.md](./02_VBM_Falsifizierbarkeit.md)
- [03_v142_Decodierbarkeit.md](./03_v142_Decodierbarkeit.md)
- [04_v143_Wortuebergreifender_Zustand.md](./04_v143_Wortuebergreifender_Zustand.md)
- [05_v145_Nahtkompatible_Codebooks.md](./05_v145_Nahtkompatible_Codebooks.md)
- [06_v146_Abkuerzungssysteme.md](./06_v146_Abkuerzungssysteme.md)
- [07_v147_Kapazitaetsgrenzen.md](./07_v147_Kapazitaetsgrenzen.md)
- [08_Schreibpraktische_Pruefung.md](./08_Schreibpraktische_Pruefung.md)
- [09_Paketindex_original.md](./09_Paketindex_original.md)

## Gesamte Arbeitsdateien – auf diesem Branch verfügbar

Die vollständige **entpackte** Sammlung liegt unter [Material](./Material/). Sie enthält 316 Einträge aus dem Archiv (Originalgenerator, Quellcode, Eingabedaten, Prüfberichte, Ergebnistabellen und Binär-Checkpoints), jeweils in ihrer Unterordnerstruktur. Die Vollständigkeit wurde anhand des Archiv-Manifests mit Dateipfaden und Größen überprüft.

**Einzige Transport-Ausnahme:** `Material/05_v145_bis_v147/VBM_v145_Seam_Compatible_Codebooks_2026-10-14/vbm_v145/inputs/voynich_transcriptions_slim.json` wird als `voynich_transcriptions_slim.json.gz` gespeichert. Diese verlustfreie gzip-Komprimierung war wegen der Upload-Grenze erforderlich. Zur Wiederherstellung `gzip -dk voynich_transcriptions_slim.json.gz` verwenden. Die rekonstruierte Datei besitzt unverändert 4.483.164 Byte und den SHA-256-Hash `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`. Siehe `inputs/RESTORE_DATA.txt`.

**Das Gesamt-ZIP ist nicht notwendig:** Es muss nicht gesondert heruntergeladen oder hochgeladen werden. Der komplette Dateiinhalt ist im Branch einzeln verfügbar.

**Vertraulichkeit:** Das Repository ist öffentlich. Joachims Originaldateien wurden in diesem Arbeitszweig als Einzeldateien veröffentlicht. Eine separate Genehmigung für eine öffentliche Weitergabe wurde in dieser Dokumentation nicht geprüft; gegebenenfalls ist der Zugriff auf das Repository einzuschränken.
