import chess
import chess.engine
import os
from pathlib import Path

current_path = Path().absolute()


class StockFishEngine:
    def __init__(self) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(
            os.path.join(current_path,
                         "..", "stockfish", "stockfish_15_x64_avx2.exe")
        )

    def move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=0.0001))
        return result.move

    def quit(self):
        self.engine.quit


stockFishEngine = StockFishEngine()
