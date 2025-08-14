# YE OLD COAD of LEEYTE
from tqdm import tqdm


def filter_type(board, points, query):
    satisfies = []
    for point in points:
        r = get(board, point)
        if r and r == query:
            satisfies.append(point)
    return satisfies


def get_dims(board):
    height = len(board)
    width = len(board[0])
    return (width, height)


def get(board, coord):
    x = coord[0]
    y = coord[1]

    width, height = get_dims(board)

    if not (0 <= x < width):
        return None
    if not (0 <= y < height):
        return None

    return board[y][x]


def is_on_boundary(coord, board):
    for nayb in moore(coord, get_dims(board)):
        thing = get(board, nayb)
        if thing and thing == ".":
            return True
    return False


def score(hamlet, board):
    #   * At least one Water. Additional Waters provide no benefit (W > 1).
    #   * At least one Game. More than two Games provide no benefit (G > 2).
    # The following squares make for better hamlets:
    #   * A Defense square, but only if it is on a boundary. The more the better.
    #   * A Cave square, but having more than one doesn’t help.
    #   * A Fertile ground square. One is great, additional are good.

    waters = filter_type(board, hamlet, "W")
    games = filter_type(board, hamlet, "G")
    caves = filter_type(board, hamlet, "C")
    defenses = filter_type(board, hamlet, "D")
    fertiles = filter_type(board, hamlet, "F")

    score = 0
    score += 0 if len(waters) == 0 else 1
    score += min(len(games), 2)

    # defense
    num_ds = 0
    for d in defenses:
        if is_on_boundary(d, board):
            num_ds += 1
    score += num_ds

    # cave
    if len(caves) >= 1:
        score += 1

    # fertiles
    score += len(fertiles)

    return score


def moore(coord, dims):
    x = coord[0]
    y = coord[1]

    naybs = []
    width, height = dims
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height:
            naybs.append((nx, ny))

    print(f"moore of {coord} is {naybs=}, dims: {dims=}")
    return naybs


full_board = [
    ["W", "G", ".", "F", ".", "W", "."],
    [".", "C", ".", "C", ".", ".", "C"],
    [".", ".", "D", ".", "G", "D", "F"],
    ["G", "F", "W", ".", "F", "W", "."],
    [".", ".", ".", "W", ".", ".", "."],
    ["G", "C", ".", "W", ".", "C", "W"],
    ["F", ".", "C", ".", "F", "D", "F"],
    [".", "W", "F", "G", ".", "G", "."],
]

mini_board = [
    ["W", "."],
    [".", "."],
]


def find_hamlets_exhaustively(board):
    distinct_hamlets = set()

    dims = get_dims(board)
    width, height = dims

    # seeds
    seed_positions = []
    for y in range(0, height):
        for x in range(0, width):
            seed_positions.append((x, y))

    print("seed positions")
    print(seed_positions)

    for x, y in tqdm(seed_positions):
        seed = (x, y)
        print(f"{seed=}")
        seed_type = get(board, seed)
        print(f"{seed_type=}")
        # root hamlets
        if seed_type == ".":
            print("skip this seed")
            continue
        else:
            seed_hamlet = (seed,)
            distinct_hamlets.add(seed_hamlet)

        print(f"{seed_hamlet=}")

        growing_hamlet = [seed]
        checked = set()
        checked.add(seed)

        search_stack = []
        naybs = moore(seed, dims)
        print(f"\t{naybs=}")
        search_stack.extend(naybs)
        while search_stack:
            print(f"\t{search_stack=}")
            nayb = search_stack.pop()
            print(f"\tpoped: {nayb}")

            if nayb in checked:
                continue
            checked.add(nayb)

            nayb_t = get(board, nayb)
            print(f"\t{nayb_t=}")

            # skip if .
            if get(board, nayb) == ".":
                print("\tskipped this nayb")
                continue
            else:
                # if not barren, add to your hamlet points list, sort, see if unique
                # # if unique, add to distinct hamlets
                # # # add new naybs of new point
                growing_hamlet.append(nayb)
                frozen_hamlet = tuple(sorted(growing_hamlet))
                if frozen_hamlet in distinct_hamlets:
                    continue
                distinct_hamlets.add(frozen_hamlet)
                new_naybs = moore(nayb, dims)

                search_stack.extend(new_naybs)

    return distinct_hamlets


def viz(hamlet, board):
    print(type(hamlet))
    print(hamlet)
    hamlet = list(hamlet)
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    first = hamlet.pop()
    x, y = first
    min_x = x
    max_x = x
    min_y = y
    max_y = y

    for p in hamlet:
        x, y = p
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    chars = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            t = get(board, (x, y))
            if t:
                chars.append(" " if t == "." else t)
        chars.append("\n")

    return "".join(chars)


if __name__ == "__main__":
    board = full_board
    all_unique_hamlets = find_hamlets_exhaustively(board)
    scores = []
    for hamlet in all_unique_hamlets:
        s = score(hamlet, board)
        scores.append((hamlet, s))

    sorted_scores = sorted(scores, key=lambda h_s: h_s[1])
    top = sorted_scores[-1]
    top_hamlet, top_hamlet_score = top
    print(viz(top_hamlet, board))
    print(f"score of {top_hamlet_score}")
