# Benutzerhandbuch: Resonanz-Äther Gravitation Simulations-Tool

_Willkommen beim Benutzerhandbuch für das Resonanz-Äther Gravitation Simulations-Tool. Diese Anwendung wurde entwickelt, um verschiedene Gravitationsmodelle zu analysieren, anzupassen und zu visualisieren, insbesondere im Kontext eines hypothetischen Resonanz-Äther-Konzepts. Sie ermöglicht es Ihnen, die Gravitationsbeschleunigung in Abhängigkeit von der Höhe zu modellieren, Fallbewegungen unter normalen und resonanten Bedingungen zu simulieren und Gravitationsfelder zu visualisieren. Das Tool löst das Problem, komplexe physikalische Modelle der Gravitation zu vergleichen und die Auswirkungen von Resonanzphänomenen auf die Fallbewegung zu untersuchen, indem es präzise numerische Anpassungen und klare grafische Darstellungen bietet._

## 1. Erste Schritte: Installation und Ausführung

Die Anwendung ist ein Python-Skript und erfordert eine Python-Umgebung sowie einige wissenschaftliche Bibliotheken. 

### 1.1 Voraussetzungen
Stellen Sie sicher, dass die folgenden Bibliotheken in Ihrer Python-Umgebung installiert sind:
*   `numpy` (für numerische Berechnungen)
*   `matplotlib` (für die Erstellung von Diagrammen)
*   `scipy` (optional, für erweiterte Optimierungsfunktionen; falls nicht vorhanden, wird ein Fallback-Algorithmus verwendet)

Sie können diese Bibliotheken über `pip` installieren:
```bash
pip install numpy matplotlib scipy
```

### 1.2 Anwendung starten
1.  **Speichern Sie das Skript:** Speichern Sie den bereitgestellten Python-Code (z.B. `code.py`) in einem Verzeichnis Ihrer Wahl.
2.  **Öffnen Sie ein Terminal/Kommandozeile:** Navigieren Sie zu dem Verzeichnis, in dem Sie das Skript gespeichert haben.
3.  **Führen Sie das Skript aus:** Geben Sie den folgenden Befehl ein und drücken Sie Enter:
    ```bash
    python code.py
    ```

### 1.3 Ausgabeordner
Nach der Ausführung erstellt die Anwendung automatisch einen Ordner namens `aether_report_out` im selben Verzeichnis, in dem sich das Skript befindet. In diesem Ordner werden alle generierten Diagramme und der Abschlussbericht gespeichert.

## 2. Die Benutzeroberfläche (CLI) im Überblick

Diese Anwendung besitzt keine grafische Benutzeroberfläche im herkömmlichen Sinne. Sie wird über die Kommandozeile (CLI) ausgeführt und kommuniziert über Textausgaben im Terminal. Die Ergebnisse werden in Form von Bilddateien und einem Markdown-Bericht im Ausgabeordner `aether_report_out` bereitgestellt.

### 2.1 Terminal-Ausgabe
Während der Ausführung sehen Sie im Terminal folgende Meldungen:
*   `Fitting ...`: Zeigt an, dass die Anwendung die verschiedenen Gravitationsmodelle an die Referenzdaten anpasst.
*   `Fertig. Erstelle Diagramme ...`: Signalisiert den Abschluss der Anpassung und den Beginn der Diagrammerstellung.
*   `=== Ergebnisse ===`: Leitet die Zusammenfassung der Anpassungsergebnisse ein.
*   Anschließend werden die RMSE-Werte (Root Mean Square Error) für jedes Modell angezeigt, z.B. `1×Exp RMSE = ...`, `2×Exp RMSE = ...`, `Hybrid RMSE = ... (beste Anpassung)`.
*   `Report & Bilder: [Pfad zum Ausgabeordner]`: Zeigt den vollständigen Pfad zum Ordner an, in dem die Ergebnisse gespeichert wurden.

### 2.2 Ausgabeordner `aether_report_out`
Dieser Ordner enthält die wichtigsten Ergebnisse der Anwendung:
*   `profiles.png`: Ein Diagramm, das den Vergleich der angepassten Gravitationsmodelle mit der Referenzkurve zeigt.
*   `drop.png`: Ein Diagramm, das die Simulation einer Fallbewegung unter normalen und resonanten Bedingungen darstellt.
*   `field.png`: Ein Diagramm, das das Gravitationspotential und die Feldlinien visualisiert.
*   `report.md`: Ein Markdown-Bericht, der eine Zusammenfassung der Ergebnisse, die Bewertung der Modelle und Links zu den generierten Diagrammen enthält.

## 3. Kernfunktionen und Workflows

Die Anwendung führt einen vordefinierten Workflow aus, der aus Modellierung, Optimierung, Simulation und Visualisierung besteht. Als Benutzer starten Sie diesen Workflow durch die Ausführung des Skripts.

### 3.1 Gravitationsmodelle anpassen
Die Anwendung passt automatisch drei verschiedene Gravitationsmodelle an eine Ziel-Gravitationskurve (die klassische 1/r²-Abhängigkeit) an:
1.  **1×Exp (Einfaches Exponentialmodell):** Ein Modell mit einem einzelnen exponentiellen Term.
2.  **2×Exp (Zweifaches Exponentialmodell):** Ein Modell mit zwei exponentiellen Termen.
3.  **Hybridmodell:** Eine Kombination aus dem klassischen 1/r²-Term und zwei exponentiellen Korrekturtermen. Dieses Modell ist darauf ausgelegt, die beste Anpassung zu erreichen.

Die Anpassung erfolgt durch Optimierung der Modellparameter, um den 'Root Mean Square Error' (RMSE) zwischen dem Modell und der Zielkurve zu minimieren. Die Ergebnisse der Anpassung (die optimalen Parameter und der erreichte RMSE) werden im Terminal ausgegeben und im Bericht dokumentiert.

### 3.2 Fallbewegung simulieren
Nach der Anpassung des Hybridmodells simuliert die Anwendung eine Fallbewegung aus einer Höhe von 10 Metern. Es werden zwei Szenarien verglichen:
*   **Normal:** Die Fallbewegung unter dem Einfluss der normalen Gravitation, wie sie durch das Hybridmodell beschrieben wird.
*   **Resonanz:** Eine hypothetische Fallbewegung, bei der ein 'Levitationsfaktor' angewendet wird, der die Gravitationswirkung neutralisiert. Dies simuliert einen Zustand, in dem ein Objekt aufgrund von Resonanzphänomenen nicht beschleunigt wird und seine Höhe beibehält.

Die Ergebnisse dieser Simulation werden im Diagramm `drop.png` visualisiert, das die Höhe und Geschwindigkeit des fallenden Objekts über die Zeit darstellt.

### 3.3 Gravitationsfelder visualisieren
Die Anwendung berechnet und visualisiert auch das Gravitationspotential und die Feldlinien des Hybridmodells in einem 2D-Schnitt. Dies bietet einen Einblick in die Struktur des modellierten Gravitationsfeldes.
*   **Potential Φ:** Zeigt die Verteilung des Gravitationspotentials im Raum an.
*   **Feldlinien:** Stellt die Richtung und Stärke der Gravitationskraft dar.

Diese Visualisierungen sind im Diagramm `field.png` zu finden.

## 4. Detaillierte Funktionsbeschreibung der Ausgaben

Die Anwendung generiert mehrere Ausgabedateien, die die Ergebnisse der Simulationen detailliert darstellen.

### 4.1 `profiles.png` – Modellvergleich
Dieses Diagramm vergleicht die Leistung der drei Gravitationsmodelle mit der Referenzkurve (Ziel 1/r²).
*   **X-Achse:** Höhe in Kilometern (Abstand von der Erdoberfläche).
*   **Y-Achse:** Gravitationsbeschleunigung g(r) in m/s².
*   **Schwarze Linie ('Ziel 1/r²'):** Die ideale Gravitationskurve, die die Modelle annähern sollen.
*   **Gestrichelte Linie ('A) 1×Exp'):** Das einfache Exponentialmodell.
*   **Gepunktete Linie ('B) 2×Exp'):** Das zweifache Exponentialmodell.
*   **Durchgezogene Linie ('C) Hybrid (best)'):** Das Hybridmodell, das die beste Anpassung an die Zielkurve bietet.

Dieses Diagramm hilft zu beurteilen, welches Modell die Gravitationsbeschleunigung am genauesten reproduziert.

### 4.2 `drop.png` – Fallsimulation
Dieses Diagramm zeigt die Ergebnisse der Fallsimulation in zwei Unterdiagrammen:

**Linkes Unterdiagramm: Höhe über Zeit**
*   **X-Achse:** Zeit in Sekunden.
*   **Y-Achse:** Höhe in Metern.
*   **Blaue Linie ('Normal'):** Die Höhe des Objekts bei einem normalen Fall unter Gravitation.
*   **Gestrichelte orange Linie ('Resonanz'):** Die Höhe des Objekts im hypothetischen Resonanzzustand, bei dem die Gravitation neutralisiert ist.

**Rechtes Unterdiagramm: Geschwindigkeit über Zeit**
*   **X-Achse:** Zeit in Sekunden.
*   **Y-Achse:** Geschwindigkeit v in m/s.
*   **Blaue Linie:** Die Geschwindigkeit des Objekts bei einem normalen Fall.
*   **Gestrichelte orange Linie:** Die Geschwindigkeit des Objekts im Resonanzzustand (bleibt bei 0).

Dieses Diagramm visualisiert den Unterschied zwischen einem normalen Fall und einem durch Resonanz beeinflussten Fall.

### 4.3 `field.png` – Feldvisualisierung
Dieses Diagramm zeigt eine 2D-Visualisierung des Gravitationspotentials und der Feldlinien:

**Linkes Unterdiagramm: Potential Φ**
*   Zeigt das Gravitationspotential als Konturdiagramm. Verschiedene Farben repräsentieren unterschiedliche Potentialwerte. Eine Farbleiste (Colorbar) am rechten Rand des Diagramms gibt die Skala der Potentialwerte an.

**Rechtes Unterdiagramm: Feldlinien**
*   Stellt die Gravitationsfeldlinien dar. Diese Linien zeigen die Richtung, in die ein Objekt beschleunigt würde, und ihre Dichte kann ein Indikator für die Feldstärke sein.

Dieses Diagramm bietet eine intuitive Darstellung des Gravitationsfeldes, das durch das Hybridmodell beschrieben wird.

### 4.4 `report.md` – Abschlussbericht
Der `report.md` ist eine Markdown-Datei, die eine Zusammenfassung aller wichtigen Ergebnisse enthält. Sie können diese Datei mit jedem Texteditor oder Markdown-Viewer öffnen.
*   **Kopfzeile:** Enthält Informationen wie Zeitstempel, das beste Modell, den RMSE-Wert und eine Gesamtbewertung.
*   **Abbildungen:** Enthält Links zu den generierten PNG-Dateien, sodass Sie die Diagramme direkt aus dem Bericht heraus betrachten können.
*   **Detaillierter Bericht (`report_expanded.md`):** Eine erweiterte Version des Berichts, die zusätzliche Informationen zu Motivation, mathematischen Grundlagen, detaillierten Ergebnissen, Schlussfolgerungen und einem Ausblick enthält. Diese Datei ist besonders nützlich für ein tieferes Verständnis der zugrunde liegenden Konzepte und der wissenschaftlichen Einordnung der Ergebnisse.

## 5. Einstellungen und Konfiguration (für fortgeschrittene Benutzer)

Die Anwendung bietet keine direkte Benutzeroberfläche für Einstellungen. Alle Konfigurationen müssen direkt im Python-Skript vorgenommen werden. Dies erfordert grundlegende Programmierkenntnisse in Python.

### 5.1 Grundeinstellungen (`0) Grundeinstellungen`)
Im Abschnitt `0) Grundeinstellungen` des Skripts können Sie grundlegende physikalische Konstanten und Simulationsbereiche anpassen:
*   `OUTDIR = "aether_report_out"`: Der Name des Ausgabeordners. Kann geändert werden, um Ergebnisse in einem anderen Verzeichnis zu speichern.
*   `g0 = 9.81`: Die Erdbeschleunigung an der Oberfläche.
*   `R = 6_371_000.0`: Der Erdradius in Metern.
*   `rmin, rmax = R, 3.0 * R`: Der Bereich des Abstands vom Erdmittelpunkt, über den die Modelle angepasst werden. `rmin` ist der Erdradius, `rmax` ist das Dreifache des Erdradius. Eine Änderung hier beeinflusst den Bereich der Diagramme.
*   `r_grid = np.linspace(rmin, rmax, 600)`: Die Anzahl der Punkte, die für die Modellierung und Anpassung verwendet werden. Eine höhere Zahl führt zu präziseren, aber langsameren Berechnungen.

### 5.2 Simulationsparameter für Fallbewegung (`simulate_drop_hybrid`)
In der Funktion `simulate_drop_hybrid` können Sie die Parameter der Fallsimulation anpassen:
*   `h0=10.0`: Die anfängliche Fallhöhe in Metern. Standardmäßig 10 Meter.
*   `dt=0.002`: Das Zeitschrittintervall für die Simulation in Sekunden. Kleinere Werte erhöhen die Genauigkeit, aber auch die Rechenzeit.
*   `tmax=6.0`: Die maximale Simulationszeit in Sekunden.
*   `fm=40, fa=40, bw=5`: Diese Parameter steuern den 'Levitationsfaktor' für die Resonanzsimulation. `fm` ist die Messfrequenz, `fa` die Resonanzfrequenz und `bw` die Bandbreite. Eine Änderung dieser Werte beeinflusst, wie stark die Gravitation im Resonanzszenario neutralisiert wird.

## 6. Fehlerbehebung und Häufig gestellte Fragen (FAQ)

Hier finden Sie Lösungen für häufig auftretende Probleme und Antworten auf Fragen zur Anwendung.

### 6.1 Fehlermeldung: `ModuleNotFoundError`
*   **Problem:** Sie erhalten eine Fehlermeldung wie `ModuleNotFoundError: No module named 'numpy'` oder `No module named 'matplotlib'`.
*   **Lösung:** Dies bedeutet, dass eine der erforderlichen Python-Bibliotheken nicht installiert ist. Stellen Sie sicher, dass Sie alle unter '1.1 Voraussetzungen' genannten Bibliotheken mit `pip install [Bibliotheksname]` installiert haben.

### 6.2 `scipy.optimize.minimize` Fehler
*   **Problem:** Die Anwendung meldet, dass `scipy` nicht verfügbar ist, oder die Optimierung scheint langsamer zu sein.
*   **Lösung:** Die Anwendung ist so konzipiert, dass sie auch ohne `scipy` funktioniert, indem sie auf einen zufälligen Suchalgorithmus zurückgreift. Dies ist normal, wenn `scipy` nicht installiert ist. Wenn Sie die schnellere und präzisere Optimierung wünschen, stellen Sie sicher, dass `scipy` installiert ist (`pip install scipy`).

### 6.3 Keine Diagramme oder Berichte im Ausgabeordner
*   **Problem:** Der Ordner `aether_report_out` wird erstellt, ist aber leer, oder es fehlen Dateien.
*   **Lösung:** Überprüfen Sie die Terminal-Ausgabe auf Fehlermeldungen. Stellen Sie sicher, dass das Skript vollständig ausgeführt wurde und keine Fehler während der Diagrammerstellung aufgetreten sind. Möglicherweise gibt es Probleme mit den Dateiberechtigungen, die das Speichern der Dateien verhindern. Versuchen Sie, das Skript in einem anderen Verzeichnis auszuführen, für das Sie Schreibrechte haben.

### 6.4 Diagramme sehen anders aus als erwartet
*   **Problem:** Die generierten Diagramme weichen von den Beispielen ab oder zeigen unerwartete Kurvenverläufe.
*   **Lösung:** Dies könnte an Änderungen in den `0) Grundeinstellungen` oder an den Simulationsparametern liegen. Überprüfen Sie, ob Sie versehentlich Werte wie `rmin`, `rmax`, `h0`, `dt` oder die Resonanzparameter geändert haben. Setzen Sie diese gegebenenfalls auf die Standardwerte zurück.

### 6.5 Wie interpretiere ich den RMSE-Wert?
*   **Frage:** Was bedeutet der RMSE-Wert in den Ergebnissen?
*   **Antwort:** RMSE steht für 'Root Mean Square Error' (Wurzel des mittleren quadratischen Fehlers). Er ist ein Maß dafür, wie gut ein Modell die Referenzdaten annähert. Ein kleinerer RMSE-Wert bedeutet eine bessere Anpassung. Die Anwendung gibt auch eine prozentuale Abweichung an, die den RMSE relativ zur Erdbeschleunigung `g0` darstellt. Ein Wert unter 2% (oder sogar < 0.001% für das Hybridmodell im erweiterten Bericht) gilt als sehr gute Anpassung.

