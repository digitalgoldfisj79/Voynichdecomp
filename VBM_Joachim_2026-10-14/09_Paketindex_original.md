# VBM – Kurzübersicht für Joachim

**Stand: 14. Oktober 2026.** Dieses Paket enthält die in dieser Unterhaltung verfügbaren Original- und Auswertungsarchive zu VBM. Die darin enthaltenen ZIP-Dateien sind unverändert übernommen; sie sollten bei Bedarf einzeln entpackt werden.

## Wichtigste Ergebnisse

- **Originaler Generator v138:** Reproduktion erfolgreich, aber die Prüfung zeigt nicht eindeutige Zellzuordnungen, Abweichungen im Zustandsübergang, unzulässige Zeichenübergänge und einen wenig trennscharfen Ähnlichkeitstest. Das ist eine Bewertung dieser Implementierung, keine Widerlegung jeder möglichen VBM.
- **v141/v142:** Neue, abgewandelte und vollständig reversible Encoder dekodieren 68.396 normalisierte Entwicklungs- und 13.232 Hartlieb-Wörter exakt. **Wichtige Abweichung:** Brücken wurden *innerhalb* der Ausgabewörter realisiert; die ausgegebenen Leerzeichen entsprechen Klartextwortgrenzen. Das ist **nicht** die ursprüngliche VBM mit Vokalbrücken über EVA-Leerzeichen hinweg.
- **v143/v144:** Wortübergreifende Zustandsübertragung lässt sich synthetisch realisieren und erkennen. Ein autonomer Prozess kann aber dieselben Ausgaben exakt reproduzieren. Damit ist keine historische Verschlüsselung identifiziert.
- **Realer Voynich-Text:** Currier A/B wurden getrennt untersucht; der vorläufige Effekt in Stars B war unter einer alternativen Repräsentation nicht robust. Andere Teilmengen waren negativ oder für die festgelegte Signalstärke zu schwach kalibriert. Insgesamt **kein positiver Mechanismus-Nachweis**.
- **v145:** Verbesserte Trigramm-Verträglichkeit bei weiterhin exakter Dekodierung, aber unpassende Wortlängen (ungefähr 8 Zeichen gegenüber ungefähr 5 im Referenztext). Oberflächenanpassung nutzte Manuskript-Referenzdaten und ist deshalb keine unabhängige historische Validierung.
- **v146:** Kleine, schreibpraktisch einfachere Alphabete und Abkürzungen verbessern die Längen, lösen aber die Formbeschränkungen nicht gleichzeitig.
- **v147:** Einschränkende mathematische Kapazitätsresultate für bestimmte feste Substitutionssysteme. Kein allgemeiner Unmöglichkeitsbeweis gegen zustandsabhängige oder abkürzende Verschlüsselung.
- **Schreibpraktische Prüfung:** Ein historisch plausibles Verfahren müsste ohne große moderne Optimierungstabellen umsetzbar sein und tatsächliche Wortlängen, Glyphenverbindungen, Zeilengestaltung und Fehlertypen erklären. Diese Gesamtprüfung ist bislang **nicht bestanden**.

## Interpretation und Einschränkungen

Die Tests belegen technische Eigenschaften **selbst entworfener Varianten**. Sie liefern weder eine Voynich-Entzifferung noch eine verlässliche Bestimmung einer zugrunde liegenden Sprache. Insbesondere kann ein rein aus sichtbarer Zeichenstruktur abgeleitetes Muster eine Verschlüsselung nicht eindeutig von einem gleich leistungsfähigen autonomen Erzeugungsverfahren unterscheiden. Die spätere Wiederverwendung der Hartlieb-Daten begrenzt deren Wert als noch völlig unberührter Testkorpus.

## Paketstruktur

- `01_Original/`: Joachim originales Prüfzip (unverändert).
- `02_Grundlagen_und_Pruefvertraege/`: Ausgangsaudit und Falsifizierbarkeitsvertrag.
- `03_v141_bis_v142/`: erste reversible und überarbeitete Architekturvarianten.
- `04_v143_bis_v144/`: wortübergreifender Zustand, alternative Erzeugung, Currier-A/B-Tests.
- `05_v145_bis_v147/`: Übergangskontrollen, Abkürzungen, beschränkte Kapazität.
- `06_Schreibpraxis/`: mathematische und schreibpraktische Machbarkeitsgrenzen.
- `07_Weitere_Diagnosen/`: ergänzende Tests zu Nähten und sichtbarer Zustandsabhängigkeit.
- `MANIFEST_SHA256.csv`: Dateiverzeichnis mit SHA-256-Prüfsummen.
- `E-Mail_an_Joachim_DE.txt`: Entwurf der Nachricht.

Die Archive enthalten die dort verfügbaren Skripte, Ergebnisse, Protokolle und Testdaten. **Nicht beansprucht wird Vollständigkeit bezüglich anderer Chatverläufe, nicht lokal zugänglicher Claude-Archive oder späterer externer Datenänderungen.** An einigen Stellen wird der ursprüngliche externe Transkriptionsstand per Quellenangabe referenziert, statt im ZIP vervielfältigt zu werden. Einzelheiten stehen in den jeweiligen Berichten.