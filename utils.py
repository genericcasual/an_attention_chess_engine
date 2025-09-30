import chess, chess.pgn
import time
import jax.numpy as jnp
from flax import nnx
import jax
import optax
import orbax.checkpoint as ocp
from etils.epath import Path
from models import *
import os
import math
import zstandard as zstd
import mmap
import struct
from collections.abc import Sequence
from typing import Any, SupportsIndex

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

PIECES_INVERTED = {
    "P": BLACK_PAWN,
    "N": BLACK_KNIGHT,
    "B": BLACK_BISHOP,
    "R": BLACK_ROOK,
    "Q": BLACK_QUEEN,
    "K": BLACK_KING,
    "p": WHITE_PAWN,
    "n": WHITE_KNIGHT,
    "b": WHITE_BISHOP,
    "r": WHITE_ROOK,
    "q": WHITE_QUEEN,
    "k": WHITE_KING,
}

WS_DIR = os.getcwd()


def fen_to_board_flattened(fen: str):
    parts = fen.split(" ")
    is_white = parts[1] == "w"

    if is_white:
        return jnp.array(fen_to_board_white(fen)).flatten()

    else:
        return jnp.array(fen_to_board_black(fen)).flatten()


def is_white_turn(fen: str):
    parts = fen.split(" ")
    return parts[1] == "w"


def fen_to_board_white(fen: str):
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


def fen_to_board_black(fen: str):
    parts = fen.split(" ")
    board_state = parts[0]
    rows = board_state.split("/")
    board = []
    for each_row in reversed(rows):
        row_val = []
        for piece in each_row:
            if piece.isdigit():
                string = [0 for i in range(int(piece))]
                row_val.extend(string)
            else:
                row_val.append(PIECES_INVERTED[piece])
        board.append(row_val)
    # print(board)
    return board


def list_of_fen_to_board_flattened(fen_list):
    input_list = [fen_to_board_flattened(fen) for fen in fen_list]

    return jnp.array(input_list)


def make_move(chess_game_logic: chess.Board, model, force_checkmates=True, debug = False):
    all_moves = list(chess_game_logic.legal_moves)
    eval_list = []
    fen_list = []
    for i in range(len(all_moves)):
        chess_game_logic.push(all_moves[i])
        if chess_game_logic.is_checkmate() and force_checkmates:
            return all_moves[i]
        fen = chess_game_logic.fen()
        fen_list.append(fen)
        chess_game_logic.pop()
    input_arr = list_of_fen_to_board_flattened(fen_list)
    evals = make_evaluation(model, input_arr)
    eval_list = list(zip(evals, all_moves))
    eval_list.sort(reverse=True, key=lambda tup: tup[0])
    if debug:
        print(eval_list)
    if not chess_game_logic.is_checkmate():
        if chess_game_logic.turn == True:  # white:
            chess_game_logic.push(eval_list[0][1])
            return eval_list[0][1]
        else:  # black
            chess_game_logic.push(eval_list[-1][1])
            # return eval_list[-1][1]
            return eval_list[0][1]
            # return max value, since make_evaluation is from current players perspective


def make_evaluation(model, board_array: jax.Array):
    return batched_model(model, board_array)


@nnx.jit
def batched_model(model, x_batch):
    return jax.vmap(model)(x_batch)


def win_prob_to_eval(win_prob):
    if win_prob < 0.00001:
        return -50.0
    if win_prob > 0.99999:
        return 50
    evaluation = (math.log(2 / (((win_prob - 0.5) / 0.5) + 1) - 1)) / -0.00368208 / 100
    return evaluation


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


class Game_Logic:
    def __init__(
        self, agent_white, agent_black, chess_board: chess.Board = chess.Board()
    ):
        self.white = agent_white
        self.black = agent_black
        self.board = chess_board
        self.fens = []

    def play_full_game(self,will_display=False):
        self.board.reset()
        # chess_game_logic.set_fen("2rnr3/3k1p2/2b2ppn/p1p2P1p/2P1P3/2K1B3/PP1R2P1/R7 b - - 1 34")
        while (
            not self.board.is_game_over(claim_draw=True)
            and not self.board.is_checkmate()
        ):
            self.play_full_move(will_display=will_display)

    def get_result(self, chess_game_logic: chess.Board):
        if chess_game_logic.outcome(claim_draw=True) != None:
            winner = chess_game_logic.outcome(claim_draw=True).winner
            if winner is False:
                result = -1
            elif winner is True:
                result = 1
            else:
                result = 0
            return result
        return 0
    
    def get_game_pgn(self):
        game = chess.pgn.Game().from_board(self.board)
        return str(game[0])
        
    def write_game_to_file(self, file_name):
        game = chess.pgn.Game().from_board(self.board)
        # print(game[0])
        with open(file_name, "a") as game_file:
            game_file.write(str(game[0]) + "\n")

    def play_full_move(self, will_display=False):
        if not self.board.is_game_over(claim_draw=True):
            white_move = make_move(self.board, self.white)
            self.fens.append(self.board.fen())
        if not self.board.is_game_over(claim_draw=True):
            black_move = make_move(self.board, self.black)
            self.fens.append(self.board.fen())
        print(self.board, "\n") if will_display else None

def training_loop(num_loops, model_1, model_2):
    game_logic = Game_Logic(agent_white=model_1, agent_black=model_2)
    results = {-1: 0, 0: 0, 1: 0}
    for i in range(num_loops):
        if i % 2 == 0:
            game_logic.play_full_game()
        else:
            game_logic.play_full_game()
                  
        game_logic.write_game_to_file("game_pgn.txt")
        
        result = game_logic.get_result()
        results[result] += 1
        list_of_fen = game_logic.fens

        input = list_of_fen_to_board_flattened(list_of_fen)
        shape = input.shape
        expected_output = jnp.ones(shape[0]) * result
        optimizer = nnx.Optimizer(model_2, optax.adam(0.01))  # reference sharing
        # optimizer.step.value = 0.3
        # loss = train_step(model_2, optimizer, input, expected_output)

        # print(f"{loss = }")
        print(f"{optimizer.step.value = }")
        # save_model(model_2, WS_DIR + "/checkpoints")
    return results


class BagFileReader(Sequence[bytes]):
    """Reader for single Bagz files."""

    def __init__(
        self,
        filename: str,
        *,
        separate_limits: bool = False,
        decompress: bool | None = None,
    ) -> None:
        """Creates a BagFileReader.

        Args:
          filename: The name of the single Bagz file to read.
          separate_limits: Whether the limits are stored in a separate file.
          decompress: Whether to decompress the records. If None, uses the file
            extension to determine whether to decompress.
        """
        if decompress or (decompress is None and filename.endswith(".bagz")):
            self._process = lambda x: zstd.decompress(x) if x else x
        else:
            self._process = lambda x: x
        self._filename = filename
        fd = os.open(filename, os.O_RDONLY)
        try:
            self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            file_size = self._records.size()
        except ValueError:
            self._records = b""
            file_size = 0
        finally:
            os.close(fd)
        if separate_limits:
            directory, name = os.path.split(filename)
            fd = os.open(os.path.join(directory, "limits." + name), os.O_RDONLY)
            try:
                self._limits = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                index_size = self._limits.size()
            except ValueError:
                self._limits = b""
                index_size = 0
            finally:
                os.close(fd)
            index_start = 0
        else:
            if 0 < file_size < 8:
                raise ValueError("Bagz file too small")
            self._limits = self._records
            if file_size:
                (index_start,) = struct.unpack("<Q", self._records[-8:])
            else:
                index_start = 0
            assert file_size >= index_start
            index_size = file_size - index_start
        assert index_size % 8 == 0
        self._num_records = index_size // 8
        self._limits_start = index_start

    def __len__(self) -> int:
        """Returns the number of records in the Bagz file."""
        return self._num_records

    def __getitem__(self, index: SupportsIndex) -> bytes:
        """Returns a record from the Bagz file."""
        i = index.__index__()
        if not 0 <= i < self._num_records:
            raise IndexError("bagz.BragReader index out of range")
        end = i * 8 + self._limits_start
        if i:
            rec_range = struct.unpack("<2q", self._limits[end - 8 : end + 8])
        else:
            rec_range = (0, *struct.unpack("<q", self._limits[end : end + 8]))
        return self._process(self._records[slice(*rec_range)])
