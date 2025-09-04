from flax import nnx
import chess, chess.pgn
import optax
import flax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import time

from utils import *
from etils.epath import Path

pychess = chess.Board()
pychess.reset()


model_1 = create_model()

model_2 = create_model()
pychess.reset()
fen = pychess.fen()
fen = list_of_fen_to_board_flattened([fen])
# print(model_1)


def play_full_game(chess_game_logic: chess.Board, white_model, black_model):
    chess_game_logic.reset()
    # chess_game_logic.set_fen("2rnr3/3k1p2/2b2ppn/p1p2P1p/2P1P3/2K1B3/PP1R2P1/R7 b - - 1 34")
    while (
        not chess_game_logic.is_game_over(claim_draw=True)
        and not chess_game_logic.is_checkmate()
    ):
        play_full_move(
            chess_game_logic,
            will_display=False,
            white=white_model,
            black=black_model,
        )


def get_result(chess_game_logic: chess.Board):
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


result = get_result(pychess)


@nnx.jit  # Automatic state management
def train_step(model, optimizer: nnx.Optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # In place updates.

    return loss


def training_loop(num_loops):
    results = {-1: 0, 0: 0, 1: 0}
    for i in range(num_loops):
        if i % 2 == 0:
            play_full_game(pychess, model_1, model_2)
        else:
            play_full_game(pychess, model_2, model_1)
        game = chess.pgn.Game().from_board(pychess)
        print(game[0])
        with open("game_pgn.txt", "a") as game_file:
            game_file.write(str(game[0]) + "\n")

        result = get_result(pychess)
        results[result] += 1
        list_of_fen = []
        for _ in range(len(pychess.move_stack)):
            list_of_fen.append(pychess.fen())
            pychess.pop()

        input = list_of_fen_to_board_flattened(list_of_fen)
        shape = input.shape
        expected_output = jnp.ones(shape[0]) * result
        # for training and updating weights in place
        optimizer = nnx.Optimizer(model_2, optax.adam(0.00001))  # reference sharing
        # optimizer.step.value = 0.3
        loss = train_step(model_2, optimizer, input, expected_output)

        print(f"{loss = }")
        print(f"{optimizer.step.value = }")
        print(f"{i=}")
        # save_model(model_2, WS_DIR + "/checkpoints")
    return results

print("beginning training:")
training_loop(1000)

save_model(model_2, WS_DIR + "/checkpoints")

pychess.reset()
fen = pychess.fen()
fen = list_of_fen_to_board_flattened([fen])
print(model_2(fen, deterministic=True))
print(model_1(fen, deterministic=True))
