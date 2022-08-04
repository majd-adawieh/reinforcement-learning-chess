import chess
from board import *
from agents.stockfish_engine import *


class ChessEnv():
    def __init__(self):
        self.board = chess.Board()
        self.stockFishEngine = StockFishEngine()
        pass

    def update_translated_borad(self, action):
        from_square = action.from_square
        to_square = action.to_square
        tmp = self.translated_board[7-from_square//8][from_square % 8].copy()
        self.translated_board[7-from_square//8][from_square %
                                                8] = self.translated_board[7-to_square//8][to_square % 8]
        self.translated_board[7-to_square//8][to_square % 8] = tmp

    def reset(self):
        self.board = chess.Board()
        self.translated_board = translate_board(self.board)
        return self.translated_board

    def step(self, action):
        reward = +1
        self.board.push(action)
        self.update_translated_borad(action)
        if self.board.is_checkmate():
            reward = 100
        else:
            action = self.stockFishEngine.move(self.board)
            self.board.push(action)
            self.update_translated_borad(action)
            if self.board.is_checkmate():
                reward = -100
        state_next = self.translated_board
        self.done = self.board.is_checkmate()
        return state_next, reward, self.done, None
