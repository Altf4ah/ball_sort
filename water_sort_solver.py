# Retry: Running the improved A* solver for the Water Sort Puzzle.
# Time limit: 90 seconds for this run.

from collections import deque, defaultdict
import heapq
import time

START = (
    (3,5,4,1),
    (4,1,6,2),
    (7,8,7,3),
    (10,9,4,5),
    (9,2,11,2),
    (12,12,9,3),
    (5,4,12,8),
    (3,1,5,10),
    (11,2,7,8),
    (6,8,6,9),
    (10,12,1,6),
    (11,10,7,11),
    tuple(),      # T13
    tuple()       # T14
)

TUBE_CAP = 4
NUM_TUBES = len(START)

def is_goal(state):
    full_single = 0
    empty = 0
    for t in state:
        if not t:
            empty += 1
        elif len(t) == TUBE_CAP and all(x == t[0] for x in t):
            full_single += 1
        else:
            return False
    return full_single == 12 and empty == 2

def contiguous_top_count(t):
    if not t: return 0
    color = t[-1]
    c = 0
    for i in range(len(t)-1, -1, -1):
        if t[i] == color:
            c += 1
        else:
            break
    return c

def possible_moves(state):
    moves = []
    for i in range(NUM_TUBES):
        src = state[i]
        if not src: continue
        src_top = src[-1]
        src_count = contiguous_top_count(src)
        for j in range(NUM_TUBES):
            if i == j: continue
            dst = state[j]
            if len(dst) >= TUBE_CAP: continue
            if dst and dst[-1] != src_top: continue
            space = TUBE_CAP - len(dst)
            if space == 0: continue
            # pruning: avoid pouring single unit into empty if other same-top exists
            if (not dst) and src_count == 1:
                found_better = False
                for k in range(NUM_TUBES):
                    if k==i or k==j: continue
                    if state[k] and state[k][-1] == src_top and len(state[k])<TUBE_CAP:
                        found_better = True
                        break
                if found_better:
                    continue
            moves.append((i,j))
    moves.sort(key=lambda pair: -min(contiguous_top_count(state[pair[0]]), TUBE_CAP - len(state[pair[1]])))
    return moves

def do_pour(state, i, j):
    state = [list(t) for t in state]
    src = state[i]
    dst = state[j]
    if not src: return None
    if dst and dst[-1] != src[-1]: return None
    amt = contiguous_top_count(tuple(src))
    space = TUBE_CAP - len(dst)
    move_amt = min(amt, space)
    if move_amt == 0: return None
    for _ in range(move_amt):
        dst.append(src.pop())
    new_state = tuple(tuple(t) for t in state)
    return new_state

def heuristic(state):
    score = 0
    colors_incomplete = set()
    for t in state:
        if not t:
            continue
        if len(t) == TUBE_CAP and all(x == t[0] for x in t):
            continue
        score += len(t) * 2
        colors_incomplete.add(t[-1])
    score += len(colors_incomplete) * 3
    return score

def a_star(start, time_limit=90, iter_limit=2000000):
    t0 = time.time()
    open_heap = []
    gscore = {start: 0}
    fscore = {start: heuristic(start)}
    heapq.heappush(open_heap, (fscore[start], start))
    parent = {start: None}
    move_from = {}
    iters = 0
    while open_heap:
        if time.time() - t0 > time_limit:
            return None, None, iters, time.time()-t0
        iters += 1
        _, current = heapq.heappop(open_heap)
        if is_goal(current):
            path = []
            s = current
            while parent[s] is not None:
                path.append(move_from[s])
                s = parent[s]
            path.reverse()
            return path, current, iters, time.time()-t0
        for (i,j) in possible_moves(current):
            nxt = do_pour(current, i, j)
            if nxt is None: continue
            tentative = gscore[current] + 1
            if nxt not in gscore or tentative < gscore[nxt]:
                gscore[nxt] = tentative
                parent[nxt] = current
                move_from[nxt] = (i+1, j+1)
                f = tentative + heuristic(nxt)
                heapq.heappush(open_heap, (f, nxt))
        if iters > iter_limit:
            return None, None, iters, time.time()-t0
    return None, None, iters, time.time()-t0

# Run solver
start_time = time.time()
path, final_state, iterations, elapsed = a_star(START, time_limit=90, iter_limit=2000000)
total_time = time.time() - start_time

if path is None:
    print("No solution found within time/iteration limits.")
    print(f"Iterations: {iterations}, elapsed: {elapsed:.2f}s, total runtime: {total_time:.2f}s")
else:
    print("Solution found! Moves count:", len(path))
    print(path)
    print("Final state:")
    for idx, t in enumerate(final_state, start=1):
        print(f"T{idx}:", t)
    print(f"Iterations: {iterations}, time: {elapsed:.2f}s, total runtime: {total_time:.2f}s")
