import numpy as np
import chess

pieces_one_hot_encoding = {
    'P_l': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P_r': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'p':   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N_l': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N_r': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'n':   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B_l': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B_r': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'b':   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'R_l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R_r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'r':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'q':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'Q':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'k':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'K':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


num2move = {}
move2num = {}

counter = 0
for from_square in range(64):
    for to_square in range(64):
        num2move[counter] = chess.Move(from_square, to_square)
        move2num[chess.Move(from_square, to_square)] = counter
        counter += 1


def translate_board(board):
    pgn = board.epd()
    tensor_board = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        row_tensor = []
        for index, thing in enumerate(row):
            if thing.isdigit():
                for i in range(0, int(thing)):
                    row_tensor.append(pieces_one_hot_encoding['.'])
            else:
                if thing not in ["P", "N", "R", "B"]:
                    row_tensor.append(pieces_one_hot_encoding[thing])
                else:
                    if(index < 4):
                        row_tensor.append(pieces_one_hot_encoding[thing+"_l"])
                    else:
                        row_tensor.append(pieces_one_hot_encoding[thing+"_r"])
        tensor_board.append(row_tensor)
    return np.array(tensor_board)


def filter_legal_moves(board, logits):
    filter_mask = np.zeros(logits.shape)
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        idx = move2num[chess.Move(from_square, to_square)]
        filter_mask[idx] = 1
    new_logits = logits*filter_mask
    return new_logits


def check_legal_move(board, move):
    legal_moves = board.legal_moves
    legal = False
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        if from_square == move.from_square and to_square == move.to_square:
            legal = True
            break

    return legal
