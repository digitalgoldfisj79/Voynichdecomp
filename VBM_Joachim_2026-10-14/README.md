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

## Originalarchive (separates ZIP)

Das **Gesamtpaket** `VBM_Gesamtpaket_fuer_Joachim_2026-10-14.zip` (6.408.563 Byte, SHA-256 `3bef2beebdeba2be78fb662b220409f51c4b2dd2ca1a53e8ba89abd4699012af`) enthält auch Joachims unveränderte Originaldateien und sämtliche verfügbaren Reproduzierbarkeits-ZIPs. **Dieses Binärarchiv ist zum Zeitpunkt dieser Veröffentlichung nicht auf den Branch hochgeladen.** Es kann über GitHubs Weboberfläche mit 'Add file → Upload files' im Branch ergänzt werden.

**Hinweis:** Dies ist ein **öffentlich sichtbares Repository**. Das Hochladen des Gesamtpakets würde auch Joachims Originalmaterial öffentlich verbreiten; dafür sollten die entsprechenden Weitergaberechte geklärt sein.
