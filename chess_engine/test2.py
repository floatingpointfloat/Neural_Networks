import chess
import chess.engine
import numpy as np
from datasets import load_dataset
from pathlib import Path


# ============================================================
# Einstellungen
# ============================================================

STOCKFISH_PATH = Path(
    r"C:\Users\Gigabyte\Desktop\stockfish\stockfish-windows-x86-64-universal.exe"
)

NUMBER_OF_POSITIONS = 200

MIN_DEPTH = 30

STOCKFISH_TIME = 0.2


# ============================================================
# Hugging-Face-Datensatz laden
# ============================================================

print("Lade Hugging-Face-Datensatz...")

dataset = load_dataset(
    "Lichess/chess-position-evaluations",
    split="train",
    streaming=True
)

print("Datensatz geladen.")


# ============================================================
# Stockfish starten
# ============================================================

engine = chess.engine.SimpleEngine.popen_uci(
    str(STOCKFISH_PATH)
)

engine.configure({
    "Threads": 1,
    "Hash": 128
})

print("Stockfish gestartet.")
print()


# ============================================================
# Positionen sammeln
# ============================================================

positions = []

seen_fens = set()

white_positions = 0
black_positions = 0

print("Suche geeignete Positionen...")
print()


for position in dataset:

    # --------------------------------------------------------
    # Tiefe überprüfen
    # --------------------------------------------------------

    depth = position.get("depth")

    if depth is None:
        continue

    if depth < MIN_DEPTH:
        continue


    # --------------------------------------------------------
    # Centipawn-Wert
    # --------------------------------------------------------

    hf_cp = position.get("cp")

    if hf_cp is None:
        continue


    # --------------------------------------------------------
    # Mate-Positionen überspringen
    # --------------------------------------------------------

    if position.get("mate") is not None:
        continue


    # --------------------------------------------------------
    # FEN
    # --------------------------------------------------------

    fen = position.get("fen")

    if fen is None:
        continue


    # --------------------------------------------------------
    # Doppelte FEN überspringen
    # --------------------------------------------------------

    if fen in seen_fens:
        continue

    seen_fens.add(fen)


    # --------------------------------------------------------
    # Brett erstellen
    # --------------------------------------------------------

    try:
        board = chess.Board(fen)

    except ValueError:
        continue


    # --------------------------------------------------------
    # Nicht zu extreme Bewertungen
    #
    # Dadurch verhindern wir, dass ein paar riesige Werte
    # unsere statistische Auswertung dominieren.
    # --------------------------------------------------------

    if abs(hf_cp) > 1000:
        continue


    # --------------------------------------------------------
    # Gleichmäßig Weiß / Schwarz sammeln
    # --------------------------------------------------------

    if board.turn == chess.WHITE:

        if white_positions >= NUMBER_OF_POSITIONS // 2:
            continue

        white_positions += 1

    else:

        if black_positions >= NUMBER_OF_POSITIONS // 2:
            continue

        black_positions += 1


    # --------------------------------------------------------
    # Position speichern
    # --------------------------------------------------------

    positions.append({
        "fen": fen,
        "hf_cp": hf_cp,
        "depth": depth,
        "turn": board.turn
    })


    # --------------------------------------------------------
    # Fertig?
    # --------------------------------------------------------

    if (
        white_positions >= NUMBER_OF_POSITIONS // 2
        and
        black_positions >= NUMBER_OF_POSITIONS // 2
    ):
        break


print()
print(
    f"{len(positions)} Positionen gefunden."
)

print(
    f"Weiß am Zug: {white_positions}"
)

print(
    f"Schwarz am Zug: {black_positions}"
)

print()


# ============================================================
# Stockfish-Bewertungen
# ============================================================

results = []

print("Vergleiche HF mit Stockfish...")
print()


for index, position in enumerate(positions, start=1):

    board = chess.Board(
        position["fen"]
    )


    # --------------------------------------------------------
    # Stockfish analysieren
    # --------------------------------------------------------

    info = engine.analyse(
        board,
        chess.engine.Limit(
            time=STOCKFISH_TIME
        )
    )


    score = info["score"]


    # --------------------------------------------------------
    # White Perspective
    # --------------------------------------------------------

    sf_white = score.pov(
        chess.WHITE
    ).score(
        mate_score=100000
    )


    # --------------------------------------------------------
    # Side-to-Move Perspective
    # --------------------------------------------------------

    sf_stm = score.pov(
        board.turn
    ).score(
        mate_score=100000
    )


    # --------------------------------------------------------
    # Ergebnis speichern
    # --------------------------------------------------------

    results.append({
        "fen": position["fen"],
        "turn": position["turn"],
        "hf_cp": position["hf_cp"],
        "sf_white": sf_white,
        "sf_stm": sf_stm,
        "depth": position["depth"]
    })


    # --------------------------------------------------------
    # Ausgabe
    # --------------------------------------------------------

    turn_name = (
        "White"
        if board.turn == chess.WHITE
        else "Black"
    )

    print(
        f"{index:3d}/{len(positions)} | "
        f"{turn_name:5s} | "
        f"HF: {position['hf_cp']:+5d} | "
        f"SF White: {sf_white:+5d} | "
        f"SF STM: {sf_stm:+5d}"
    )


# ============================================================
# Stockfish beenden
# ============================================================

engine.quit()

print()
print("Stockfish beendet.")
print()


# ============================================================
# Arrays erstellen
# ============================================================

hf = np.array(
    [
        result["hf_cp"]
        for result in results
    ],
    dtype=float
)

sf_white = np.array(
    [
        result["sf_white"]
        for result in results
    ],
    dtype=float
)

sf_stm = np.array(
    [
        result["sf_stm"]
        for result in results
    ],
    dtype=float
)


# ============================================================
# Korrelation
# ============================================================

correlation_white = np.corrcoef(
    hf,
    sf_white
)[0, 1]

correlation_stm = np.corrcoef(
    hf,
    sf_stm
)[0, 1]


# ============================================================
# Mittlerer absoluter Fehler
# ============================================================

mae_white = np.mean(
    np.abs(
        hf - sf_white
    )
)

mae_stm = np.mean(
    np.abs(
        hf - sf_stm
    )
)


# ============================================================
# Vorzeichenvergleich
# ============================================================

sign_white = np.sign(
    hf
) == np.sign(
    sf_white
)

sign_stm = np.sign(
    hf
) == np.sign(
    sf_stm
)


agreement_white = np.mean(
    sign_white
)

agreement_stm = np.mean(
    sign_stm
)


# ============================================================
# Ergebnisse
# ============================================================

print("=" * 65)
print("ERGEBNIS")
print("=" * 65)

print()

print(
    f"Positionen: {len(results)}"
)

print()

print(
    "Korrelation:"
)

print(
    f"  HF ↔ Stockfish White POV:      "
    f"{correlation_white:.4f}"
)

print(
    f"  HF ↔ Stockfish Side-to-Move:   "
    f"{correlation_stm:.4f}"
)

print()

print(
    "Mittlerer absoluter Fehler:"
)

print(
    f"  HF ↔ Stockfish White POV:      "
    f"{mae_white:.1f} cp"
)

print(
    f"  HF ↔ Stockfish Side-to-Move:   "
    f"{mae_stm:.1f} cp"
)

print()

print(
    "Vorzeichen-Übereinstimmung:"
)

print(
    f"  HF ↔ Stockfish White POV:      "
    f"{agreement_white * 100:.1f}%"
)

print(
    f"  HF ↔ Stockfish Side-to-Move:   "
    f"{agreement_stm * 100:.1f}%"
)


# ============================================================
# Entscheidung
# ============================================================

print()
print("=" * 65)
print("WAHRSCHEINLICHE PERSPEKTIVE")
print("=" * 65)

print()

if (
    correlation_white > correlation_stm
    and
    mae_white < mae_stm
):

    print(
        "→ Die HF cp-Werte sind sehr wahrscheinlich "
        "aus Sicht von WEISS."
    )

elif (
    correlation_stm > correlation_white
    and
    mae_stm < mae_white
):

    print(
        "→ Die HF cp-Werte sind sehr wahrscheinlich "
        "aus Sicht des SPIELERS AM ZUG."
    )

else:

    print(
        "→ Kein eindeutiges Ergebnis."
    )

print()


# ============================================================
# Einige Beispiele
# ============================================================

print("=" * 65)
print("BEISPIELE")
print("=" * 65)

print()


for result in results[:20]:

    turn_name = (
        "White"
        if result["turn"] == chess.WHITE
        else "Black"
    )

    print(
        f"{turn_name:5s} | "
        f"HF: {result['hf_cp']:+5d} | "
        f"SF White: {result['sf_white']:+5d} | "
        f"SF STM: {result['sf_stm']:+5d} | "
        f"Depth: {result['depth']}"
    )