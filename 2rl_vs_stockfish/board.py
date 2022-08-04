from matplotlib.pyplot import pie
import numpy as np
import chess

chess_dict = {
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
for from_sq in range(64):
    for to_sq in range(64):
        num2move[counter] = chess.Move(from_sq, to_sq)
        move2num[chess.Move(from_sq, to_sq)] = counter
        counter += 1


def translate_board(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for index, thing in enumerate(row):
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(chess_dict['.'])
            else:
                if thing not in ["P", "N", "R", "B"]:
                    foo2.append(chess_dict[thing])
                else:
                    if(index < 4):
                        foo2.append(chess_dict[thing+"_l"])
                    else:
                        foo2.append(chess_dict[thing+"_r"])
        foo.append(foo2)
    return np.array(foo)


def find_piece_key(piece_representation):
    index = np.argmax(piece_representation)
    piece_key = ""
    for key in chess_dict:
        if index == np.argmax(chess_dict[key]):
            piece_key = key
            break
    return piece_key


def can_move(move, agent_num, translated_board):
    from_square = move.from_square
    piece = translated_board[7-from_square//8][from_square % 8]
    piece_key = find_piece_key(piece)

    if agent_num == 0 and "_l" in piece_key:
        return True
    if agent_num == 1 and "_r" in piece_key:
        return True
    if "Q" in piece_key or "K" in piece_key:
        return True
    return False


def filter_legal_moves(board, logits, agent_num, translated_board):
    filter_mask = np.zeros(logits.shape)

    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        if(can_move(legal_move, agent_num, translated_board)):
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
