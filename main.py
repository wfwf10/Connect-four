import numpy as np
import math
import sys
from monte_carlo_tree import move, score, monte_carlo
from keras.models import load_model
import time
start_time = time.time()

# np.random.seed(2)
# tf.set_random_seed(2)

random_move_probability = 0.2  # small chance that disc drop to other columns
# TODO: trained CNN sets including [R, C, win_num]: ([8,10,5]  [5,10,4]), [6,7,4], [10,5,4], [5,5,4], [6,7,3], [5,5,3]
R = 6   # 6  # number of rows (board height)
C = 7  # 7  # number of columns (board width)
win_num = 4  # 4  # number of symbol in line to win
if win_num > C or win_num > R:
    raise ValueError('win_num is larger than board dimension!')
save_title = "R_" + str(R) + "_C_" + str(C) +"_win_num_" + str(win_num) + "_"
model = load_model('CNN_model/' + save_title + 'model.h5')
BLANK = "_"
state = np.full((R, C), BLANK)
symbol = "x"  # second player "o"
move_row = 0
move_col = 0

# TODO: Available players: "monte_carlo", "random", "human", "CNN"
player1 = "monte_carlo"
player2 = "random"


def monte_carlo_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"  # if "o" or "_"(0th turn)
    v, total_depth, move_col = monte_carlo(state, symbol, parent_row, parent_col, R, C, win_num)
    return move_col, symbol, total_depth


def random_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    move_col = valid_moves[np.random.choice(valid_moves.shape[0])]
    return move_col, symbol


def human_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    for i in range(100000):
        move_col = int(input("You play '" + symbol + "', please choose a column from: " + np.array2string(valid_moves)))
        if np.isin(move_col, valid_moves): break
        else: print("Your choice is not valid, please re-enter!")
    return move_col, symbol


def CNN_player(state, parent_row, parent_col):
    if state[parent_row][parent_col] == "x": symbol = "o"
    else: symbol = "x"  # if "o" or "_"(0th turn)
    CNN_state = np.zeros(state.shape)
    if symbol == "o": sym = -1
    else: sym = 1
    CNN_state[np.where(state == "x")] = sym
    CNN_state[np.where(state == "o")] = -sym
    CNN_state = CNN_state.reshape(1, CNN_state.shape[0], CNN_state.shape[1], 1)
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    total_nodes = 0
    flag = np.zeros(valid_moves.shape[0])
    for cnn_i, cnn_col in enumerate(valid_moves):
        cnn_child, cnn_row = move(state, symbol, cnn_col, R, C, win_num)
        v = score(cnn_child, cnn_row, cnn_col, R, C, win_num)
        total_nodes += 1
        if v in [-1, 1]:
            flag[cnn_i] = 1
            break
        cnn_valid_moves = np.nonzero(cnn_child[0][:] == BLANK)[0]
        if symbol == "x": opp_symbol = "o"
        else: opp_symbol = "x"
        for opp_col in cnn_valid_moves:
            opp_child, opp_row = move(cnn_child, opp_symbol, opp_col, R, C, win_num)
            v = score(opp_child, opp_row, opp_col, R, C, win_num)
            total_nodes += 1
            if v in [-1, 1]:
                flag[cnn_i] = -1
                break
    if flag[cnn_i] == 1:
        move_col = cnn_col
    else:
        if np.mean(flag) > -1:
            valid_moves = np.delete(valid_moves, np.argwhere(flag==-1))
        move_col = valid_moves[np.argmax(model.predict(CNN_state)[0][valid_moves])]
    return move_col, symbol, total_nodes


def random_move(state, symbol, move_col):
    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    valid_moves = np.delete(valid_moves, np.argwhere(valid_moves == move_col))
    if np.random.random() < random_move_probability and valid_moves.shape[0] > 0:
        move_col = valid_moves[np.random.choice(valid_moves.shape[0])]
        print("Go to the random column!")
    print("'" + symbol + "' played at", move_col)
    child, move_row = move(state, symbol, move_col, R, C, win_num)
    return child, move_col, move_row


def print_col_coordinate():
    pstring = "   "
    for i in range(C): pstring = pstring + str(i) + "   "
    print(pstring)


total_game_repetition = 100
player_total_nodes = 0
win_rate = 0

for game_repetition in range(total_game_repetition):
    state = np.full((R, C), BLANK)
    symbol = "x"  # second player "o"
    move_row = 0
    move_col = 0
    for turn in range(math.ceil(R*C)):
        if turn % 2 == 0:
            player = player1
        else:
            player = player2
        print("\n", game_repetition, "th game, turn", turn, ".", player, "player's turn")

        if player == "monte_carlo":
            print("Thinking...")
            move_col, symbol, total_nodes = monte_carlo_player(state, move_row, move_col)
            player_total_nodes += total_nodes
        elif player == "random":
            move_col, symbol = random_player(state, move_row, move_col)
        elif player == "human":
            move_col, symbol = human_player(state, move_row, move_col)
        elif player == "CNN":
            move_col, symbol, total_nodes = CNN_player(state, move_row, move_col)
            player_total_nodes += total_nodes
        state, move_col, move_row = random_move(state, symbol, move_col)
        print(state)
        print_col_coordinate()
        print("'" + symbol + "' played at", move_col)

        v = score(state, move_row, move_col, R, C, win_num)
        if v in [-1, 1]:
            print("\nMatch end, " + player + " with '" + symbol + "' wins!")
            if turn % 2 == 0:  # player 1
                win_rate += 1 / total_game_repetition
            break
        elif (state != BLANK).all():
            print("\nMatch end, draw!")
            if turn % 2 == 0:  # player 1
                win_rate += 0.5 / total_game_repetition
            break

print("\n\n--- %s seconds ---" % (time.time() - start_time))
print("Player ", player1, "win rate is:", win_rate)
print("The total number of nodes in tree search is:", player_total_nodes)

