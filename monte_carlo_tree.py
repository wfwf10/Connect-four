import numpy as np
import copy as cp

BLANK = "_"


def state_str(state, prefix=""):
    return "\n".join("%s%s" % (prefix, "".join(row)) for row in state)


def move(state, symbol, col, R, C, win_num):
    if state[0][col] != BLANK: return False
    new_state = cp.deepcopy(state)
    for row in range(R-1, -1, -1):
        if new_state[row][col] == BLANK:
            new_state[row][col] = symbol
            break
    return new_state, row


def score(state, row, col, R, C, win_num):
    """
    Determine the score for the state as [row][col] is played:
    +1 if player "x" has a winning line
    -1 if player "o" has a winning line
    0 otherwise
    """
    if state[row][col] == BLANK: return False
    symbol = state[row][col]
    if symbol == "x": point = 1
    else: point = -1
    # columns
    count = 0
    for r in np.arange(np.amax([0, row-win_num]), np.amin([R, row+win_num])):
        if state[r][col] == symbol:
            count += 1
            if count >= win_num: return point
        else: count = 0
    # rows
    count = 0
    for c in np.arange(np.amax([0, col-win_num]), np.amin([C, col+win_num])):
        if state[row][c] == symbol:
            count += 1
            if count >= win_num: return point
        else: count = 0
    # diagonals
    count = 0
    for i in np.arange(-1 * np.amin([row-0, col-0, win_num]), np.amin([R-row, C-col, win_num])):
        if state[row+i][col+i] == symbol:
            count += 1
            if count >= win_num: return point
        else: count = 0
    # anti diagonals
    count = 0
    for i in np.arange(-1 * np.amin([row - 0, C - col - 1, win_num]), np.amin([R - row, col + 1, win_num])):
        if state[row + i][col - i] == symbol:
            count += 1
            if count >= win_num: return point
        else:
            count = 0
    # nothing happens
    return 0


def monte_carlo(state, symbol, parent_row, parent_col, R, C, win_num):
    # 1st step of Monte-Carlo search for connect 4
    search_step = 1000  # todo: 1000 steps for CNN training data, 100 steps for single testing
    if state[parent_row, parent_col] == BLANK: v = 0
    else: v = score(state, parent_row, parent_col, R, C, win_num)
    if v in [-1, 1] or (state != BLANK).all(): return v, 1, []

    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    v, n = np.zeros(valid_moves.size), np.zeros(valid_moves.size)
    # print(valid_moves)
    for i_move, col in enumerate(valid_moves):
        child, row = move(state, symbol, col, R, C, win_num)
        # print("new iteration at col, row:", col, row)
        for iteration in range(search_step):
            v_c, n_c = search(child, 1, row, col, R, C, win_num)
            v[i_move] += v_c / search_step
            n[i_move] += n_c

    if symbol is "x": best = np.argmax(v)
    else: best = np.argmin(v)
    # print(v,n,valid_moves,best)
    return v[best], np.sum(n), valid_moves[best]


def search(state, depth, parent_row, parent_col, R, C, win_num):
    # Nesting Monte-Carlo search
    if depth >= 2*win_num: return 0, depth
    if state[parent_row][parent_col] == BLANK: return False
    elif state[parent_row][parent_col] == "o": symbol = "x"
    else: symbol = "o"
    v = score(state, parent_row, parent_col, R, C, win_num)
    if v in [-1, 1] or (state != BLANK).all(): return v, depth
    if v is False:
        print(state)
        print(parent_col, parent_row, v, depth, symbol)
        raise ValueError('invalid parent move!')

    valid_moves = np.nonzero(state[0][:] == BLANK)[0]
    col = valid_moves[np.random.choice(valid_moves.shape[0])]
    child, row = move(state, symbol, col, R, C, win_num)
    # print(col, row, v, depth, symbol)
    # print(child)
    v_c, n_c = search(child, depth+1, row, col, R, C, win_num)
    # print("search end! score: ", v_c)
    return v_c, n_c


if __name__ == "__main__":
    np.random.seed(2)
    # tf.set_random_seed(2)

    R = 6  # number of rows (board height)
    C = 7  # number of columns (board width)
    win_num = 4  # number of symbol in line to win
    if win_num > C or win_num > R:
        raise ValueError('win_num is larger than board dimension!')

    BLANK = "_"
    state = np.full((R, C), BLANK)
    symbol = "x"
    #
    # for col in range(1,6):
    #     state, row = move(state, "x", col, R, C, win_num)
    #     print(state)
    #     print(score(state, row, col, R, C, win_num))
    #
    # for i in range(5):
    #     state, row = move(state, "x", 3, R, C, win_num)
    #     print(state)
    #     print(score(state, row, 3, R, C, win_num))
    #     print(row)
    col = 4
    state, row = move(state, "x", col, R, C, win_num)

    v, depth, move_col = monte_carlo(state, "o", row, col, R, C, win_num)
    child, move_row = move(state, "o", move_col, R, C, win_num)
    print(child)
    print(depth)
    print(v, move_row, move_col)

