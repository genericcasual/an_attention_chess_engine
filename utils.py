import chess
import time
import jax.numpy as jnp
from flax import nnx
import jax
import orbax.checkpoint as ocp
from etils.epath import Path
from models import *
import os
import math


WHITE_PAWN: int = 1
WHITE_KNIGHT: int = 2
WHITE_BISHOP: int = 3
WHITE_ROOK: int = 4
WHITE_QUEEN: int = 5
WHITE_KING: int = 6
BLACK_PAWN: int = 7
BLACK_KNIGHT: int = 8
BLACK_BISHOP: int = 9
BLACK_ROOK: int = 10
BLACK_QUEEN: int = 11
BLACK_KING: int = 12

PIECES = {
    "p": BLACK_PAWN,
    "n": BLACK_KNIGHT,
    "b": BLACK_BISHOP,
    "r": BLACK_ROOK,
    "q": BLACK_QUEEN,
    "k": BLACK_KING,
    "P": WHITE_PAWN,
    "N": WHITE_KNIGHT,
    "B": WHITE_BISHOP,
    "R": WHITE_ROOK,
    "Q": WHITE_QUEEN,
    "K": WHITE_KING,
}

WS_DIR = os.getcwd()


def fen_to_board(fen: str):
    parts = fen.split(" ")
    board_state = parts[0]
    rows = board_state.split("/")
    board = []
    for each_row in rows:
        row_val = []
        for piece in each_row:
            if piece.isdigit():
                string = [0 for i in range(int(piece))]
                row_val.extend(string)
            else:
                row_val.append(PIECES[piece])
        board.append(row_val)
    # print(board)
    return board


def fen_to_board_flattened(fen: str):
    return jnp.array(fen_to_board(fen)).flatten()


def list_of_fen_to_board_flattened(fen_list):
    return jnp.array([fen_to_board_flattened(fen) for fen in fen_list])


def make_move(chess_game_logic: chess.Board, model, force_checkmates=True):
    start_time = time.time()
    all_moves = list(chess_game_logic.legal_moves)
    eval_list = []
    board_list = []
    for i in range(len(all_moves)):
        chess_game_logic.push(all_moves[i])
        if chess_game_logic.is_checkmate() and force_checkmates:
            return all_moves[i]
        fen = chess_game_logic.fen()
        board = fen_to_board_flattened(fen)
        chess_game_logic.pop()
        board_list.append(board)
        # print(board_list)
    # print(time.time() - start_time)
    board_list = jnp.array(board_list)
    evals = make_evaluation(model, board_list)
    # print(len(evals),len(all_moves))
    # print(time.time() - start_time)
    eval_list = list(zip(evals, all_moves))

    # print(time.time() - start_time)

    eval_list.sort(reverse=True, key=lambda tup: tup[0])
    # print(time.time() - start_time)
    if not chess_game_logic.is_checkmate():
        if chess_game_logic.turn == True:  # white:
            # print(eval_list[0])
            chess_game_logic.push(eval_list[0][1])
            return eval_list[0][1]
        else:  # black
            # print(eval_list[-1])
            chess_game_logic.push(eval_list[-1][1])
            return eval_list[-1][1]
    # print(time.time() - start_time)
    # return chess_game_logic.fen()


def play_full_move(
    chess_game_logic: chess.Board,
    will_display=False,
    white: Model = None,
    black: Model = None,
):  # 17 times faster
    if not chess_game_logic.is_game_over(claim_draw=True):
        white_move = make_move(chess_game_logic, white)
    if not chess_game_logic.is_game_over(claim_draw=True):
        black_move = make_move(chess_game_logic, black)
    print(chess_game_logic, "\n") if will_display else None


def make_evaluation(model, board_state_list):
    # print(board_state_list)
    arr = jnp.array([i for i in jnp.array(board_state_list)])
    y = model(x=arr)
    return y


def win_prob_to_eval(win_prob):
    if win_prob < 0.00001:
        return -50.0
    if win_prob > 0.99999:
        return 50
    evaluation = (math.log(2 / (((win_prob - 0.5) / 0.5) + 1) - 1)) / -0.00368208 / 100
    return evaluation

def write_game_to_file(chess_game_logic,file_name):
    game = chess.pgn.Game().from_board(chess_game_logic)
    # print(game[0])
    with open(file_name, "a") as game_file:
        game_file.write(str(game[0]) + "\n")


def save_model(model, directory: str):
    """
    saves the nn model to the directory,
    overwrites directory
    Args:
        model (nnx): _description_
        directory (str): ws_dir + "/checkpoints"
    """

    keys, state = nnx.state(model, nnx.RngKey, ...)
    keys = jax.tree.map(jax.random.key_data, keys)
    # nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    ocp.test_utils.erase_and_create_empty(directory)

    checkpointer.save(Path(directory) / "state", state)
    checkpointer.save(Path(directory) / "keys", keys)
    time.sleep(1)


def restore_model(checkpoint_dir: str):
    checkpointer = ocp.StandardCheckpointer()
    dir_path = Path(checkpoint_dir)
    # if not dir_path.exists():
    #     raise Exception("cannot restore model from empty directory")
    abstract_model = create_model()
    abstract_keys, abstract_state = nnx.state(abstract_model, nnx.RngKey, ...)
    abstract_keys = jax.tree.map(jax.random.key_data, abstract_keys)
    state_restored = checkpointer.restore(dir_path / "state", abstract_state)
    keys_restored = checkpointer.restore(dir_path / "keys", abstract_keys)
    nnx.update(abstract_model, keys_restored, state_restored)
    return abstract_model


