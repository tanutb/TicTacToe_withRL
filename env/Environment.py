from functools import lru_cache

# The board is a 9 character string, read left to right, top to bottom.
# '0' = empty, '1' = X, '2' = O
EMPTY_BOARD = "000000000"

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


# these two get called millions of times while training, so cache them
@lru_cache(maxsize=None)
def legal_actions(state):
    return tuple(i for i, cell in enumerate(state) if cell == "0")


@lru_cache(maxsize=None)
def winner(state):
    for a, b, c in WIN_LINES:
        if state[a] != "0" and state[a] == state[b] == state[c]:
            return state[a]
    return None


def is_terminal(state):
    return winner(state) is not None or "0" not in state


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = EMPTY_BOARD
        self.current_player = "X"
        return self.state

    def get_state(self):
        return self.state

    @property
    def board(self):
        marks = {"0": " ", "1": "X", "2": "O"}
        return [[marks[c] for c in self.state[i:i + 3]] for i in (0, 3, 6)]

    def print_board(self):
        print("\n---------\n".join(" | ".join(row) for row in self.board))

    def change_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        return {"1": "X", "2": "O"}.get(winner(self.state))

    def is_board_full(self):
        return "0" not in self.state

    def step(self, action):
        """Play (row, col) for the current player. Returns reward, next_state, done.

        The mover gets +1 for a win and 0 otherwise. The trainer gives the
        loser its -1. step() does not switch players, call change_player().
        """
        row, col = action
        if is_terminal(self.state):
            raise ValueError("The game is over, call reset() first")
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError(f"Cell {action} is off the board")
        index = row * 3 + col
        if self.state[index] != "0":
            raise ValueError(f"Cell {action} is already taken")

        mark = "1" if self.current_player == "X" else "2"
        self.state = self.state[:index] + mark + self.state[index + 1:]
        won = winner(self.state) is not None
        return float(won), self.state, won or self.is_board_full()
