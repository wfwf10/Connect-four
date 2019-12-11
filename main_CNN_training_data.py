import numpy as np
from monte_carlo_tree import move, score, monte_carlo
import time
start_time = time.time()

# np.random.seed(2)
# tf.set_random_seed(2)

random_move_probability = 0.5  # todo: modify probability!
# small chance that disc drop to other columns, intentionally set high and won't save random move as training data
R = 6   # 6  # number of rows (board height)
C = 7  # 7  # number of columns (board width)
win_num = 4  # 4  # number of symbol in line to win
if win_num > C or win_num > R:
    raise ValueError('win_num is larger than board dimension!')
BLANK = "_"

save_title = "R_" + str(R) + "_C_" + str(C) +"_win_num_" + str(win_num) + "_"
save_data = []
save_col = []


def monte_carlo_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"  # if "o" or "_"(0th turn)
    v, depth, move_col = monte_carlo(state, symbol, parent_row, parent_col, R, C, win_num)
    return move_row, move_col, symbol


def random_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    move_col = valid_moves[np.random.choice(valid_moves.shape[0])]
    return move_row, move_col, symbol


def human_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    for i in range(100000):
        move_col = int(input("You play '" + symbol + "', please choose a column from: " + np.array2string(valid_moves)))
        if np.isin(move_col, valid_moves): break
        else: print("Your choice is not valid, please re-enter!")
    return move_row, move_col, symbol


def random_move_save(state, symbol, move_col, player):
    if player == "monte_carlo":
        save_state = np.zeros(state.shape)
        if symbol == "o": sym = -1
        else: sym = 1
        save_state[np.where(state == "x")] = sym
        save_state[np.where(state == "o")] = -sym
        save_data.append(save_state)
        save_col.append(move_col)
    if np.random.random() < random_move_probability:
        valid_moves = np.nonzero(state[0][:] == BLANK)[0]
        valid_moves = np.delete(valid_moves, np.argwhere(valid_moves == move_col))
        if valid_moves.shape[0] != 0:
            move_col = valid_moves[np.random.choice(valid_moves.shape[0])]
            print("Go to the random column!")
    child, move_row = move(state, symbol, move_col, R, C, win_num)
    return child, move_col, move_row


def print_col_coordinate():
    pstring = "   "
    for i in range(C): pstring = pstring + str(i) + "   "
    print(pstring)


player1 = "monte_carlo"
player2 = "monte_carlo"

for game_repetition in range(100):
    state = np.full((R, C), BLANK)
    symbol = "x"  # second player "o"
    move_row = 0
    move_col = 0
    for turn in range(R*C):
        if turn % 2 == 0:
            player = player1
        else:
            player = player2
        print("\n", game_repetition, "th game, turn", turn, ".", player, "player's turn")

        if player == "monte_carlo":
            print("Thinking...")
            move_row, move_col, symbol = monte_carlo_player(state, move_row, move_col)
        elif player == "random":
            move_row, move_col, symbol = random_player(state, move_row, move_col)
        elif player == "human":
            move_row, move_col, symbol = human_player(state, move_row, move_col)
        state, move_col, move_row = random_move_save(state, symbol, move_col, player)
        print(state)
        print_col_coordinate()
        print("'" + symbol + "' played at", move_col)

        v = score(state, move_row, move_col, R, C, win_num)
        if v in [-1, 1]:
            print("\nMatch end, " + player + " with '" + symbol + "' wins!")
            break
        elif (state != BLANK).all():
            print("\nMatch end, draw!")
            break

np.save('CNN_data/' + save_title + 'data.npy', save_data)
np.save('CNN_data/' + save_title + 'col.npy', save_col)
# print(np.load('CNN_data/' + save_title + 'data.npy'))
# print(np.load('CNN_data/' + save_title + 'col.npy'))
print("--- %s seconds ---" % (time.time() - start_time))
