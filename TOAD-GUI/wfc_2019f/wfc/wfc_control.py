from __future__ import annotations

import datetime
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple
from .wfc_tiles import make_tile_catalog, make_mario_catalog
from .wfc_patterns import (
    pattern_grid_to_tiles,
    make_pattern_catalog_with_rotations,
)
from .wfc_adjacency import adjacency_extraction
from .wfc_solver import (
    run,
    makeWave,
    makeAdj,
    lexicalLocationHeuristic,
    lexicalPatternHeuristic,
    makeWeightedPatternHeuristic,
    Contradiction,
    StopEarly,
    makeEntropyLocationHeuristic,
    make_global_use_all_patterns,
    makeRandomLocationHeuristic,
    makeRandomPatternHeuristic,
    TimedOut,
    simpleLocationHeuristic,
    makeSpiralLocationHeuristic,
    makeHilbertLocationHeuristic,
    makeAntiEntropyLocationHeuristic,
    makeRarestPatternHeuristic,
)
from .wfc_visualize import (
    figure_list_of_tiles,
    figure_false_color_tile_grid,
    figure_pattern_catalog,
    render_tiles_to_output,
    figure_adjacencies,
    make_solver_visualizers,
    make_solver_loggers,
    tile_grid_to_image,
    save_ascii_solution,
)
import imageio.v2 as imageio
import numpy as np
import time
import logging
from numpy.typing import NDArray
import threading

logger = logging.getLogger(__name__)


def visualize_tiles(unique_tiles, tile_catalog, tile_grid):
    if False:
        figure_list_of_tiles(unique_tiles, tile_catalog)
        figure_false_color_tile_grid(tile_grid)


def visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width):
    if False:
        figure_pattern_catalog(
            pattern_catalog, tile_catalog, pattern_weights, pattern_width
        )


def make_log_stats() -> Callable[[Dict[str, Any], str], None]:
    log_line = 0

    def log_stats(stats: Dict[str, Any], filename: str) -> None:
        nonlocal log_line
        if stats:
            log_line += 1
            with open(filename, "a", encoding="utf_8") as logf:
                if log_line < 2:
                    for s in stats.keys():
                        print(str(s), end="\t", file=logf)
                    print("", file=logf)
                for s in stats.keys():
                    print(str(stats[s]), end="\t", file=logf)
                print("", file=logf)

    return log_stats


def set_ground_sky(pattern_grid, encoding, ground, sky):
    ### Ground and Sky ###
    ''' Ground defines all patterns currently at Ground level edge which is determined by direction: e.g. 3 = Down
        If Sky is set, opposite edge is fixed too
    '''
    # TODO implement ground and sky to be on the sides for other than mario levels
    ground_list: Optional[NDArray[np.int64]] = None
    if ground == 3:
        ground_list = np.vectorize(lambda x: encoding[x])(
            pattern_grid[len(pattern_grid) - 1, :]
        )
        ground_list = set(ground_list)
        if len(ground_list) == 0:
            ground_list = None
        if sky:
            sky_list = np.vectorize(lambda x: encoding[x])(
                pattern_grid[0, :]
            )
            sky_list = set(sky_list)
            if len(sky_list) == 0:
                sky_list = None
    if ground == 1:
        ground_list = np.vectorize(lambda x: encoding[x])(
            pattern_grid[0, :]
        )
        ground_list = set(ground_list)
        if len(ground_list) == 0:
            ground_list = None
        if sky:
            sky_list = np.vectorize(lambda x: encoding[x])(
                pattern_grid[len(pattern_grid) - 1, :]
            )
            sky_list = set(sky_list)
            if len(sky_list) == 0:
                sky_list = None
    return ground_list, sky_list


class CustomThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self._result = None

    def run(self):
        # Der Thread führt die Verarbeitung durch und speichert das Ergebnis in _result
        self._result = self._target(*self._args)

    def result(self):
        return self._result


def execute_wfc(
    filename: Optional[str] = None,
    tile_size: int = 1,
    pattern_width: int = 2,
    rotations: int = 8,
    output_size: Tuple[int, int] = (48, 48),
    ground: Optional[int] = None,
    sky: Optional[bool] = None,
    attempt_limit: int = 10,
    output_periodic: bool = True,
    input_periodic: bool = True,
    loc_heuristic: Literal["lexical", "hilbert", "spiral", "entropy", "anti-entropy", "simple", "random"] = "entropy",
    choice_heuristic: Literal["lexical", "rarest", "weighted", "random"] = "weighted",
    visualize: bool = False,
    global_constraint: Literal[False, "allpatterns"] = False,
    backtracking: bool = False,
    log_filename: str = "log",
    logging: bool = False,
    global_constraints: None = None,
    log_stats_to_output: Optional[Callable[[Dict[str, Any], str], None]] = None,
    *,
    image: Optional[NDArray[np.integer]] = None,
    output_destination = r"./output/",
    input_folder = r"./images/samples/",
    mario_version = False,
    ascii_file = None,
    bounds = None,
    fix_outer_bounds = False,
    difficulty_list = None
) -> NDArray[np.integer]:
    timecode = datetime.datetime.now().isoformat().replace(":", ".")
    time_begin = time.perf_counter()

    rotations -= 1  # change to zero-based

    input_stats = {
        "filename": str(filename),
        "tile_size": tile_size,
        "pattern_width": pattern_width,
        "rotations": rotations,
        "output_size": output_size,
        "ground": ground,
        "attempt_limit": attempt_limit,
        "output_periodic": output_periodic,
        "input_periodic": input_periodic,
        "location heuristic": loc_heuristic,
        "choice heuristic": choice_heuristic,
        "global constraint": global_constraint,
        "backtracking": backtracking,
    }
    # TODO Implement Mario Token Image Catalog
    if mario_version:
        visualize = False

    enable_difficulty_suggestion = False
    if difficulty_list is not None and len(difficulty_list) > 0:
        enable_difficulty_suggestion = True

    # Load the image
    if filename and not mario_version:
        if image is not None:
            raise TypeError("Only filename or image can be provided, not both.")
        image = imageio.imread(os.path.join(input_folder,  filename) + ".png")[:, :, :3]  # TODO: handle alpha channels

    if image is None and not mario_version:
        raise TypeError("An image must be given.")

    fixation = None
    if ascii_file is None and mario_version:
        raise TypeError("An ascii file must be given.")
    elif bounds is not None and mario_version:
        if bounds[0] < 0 or bounds[0] >= len(ascii_file[0]):
            raise TypeError("Starting bounds out of Boundary")
        if bounds[1] < 0 or bounds[1] >= len(ascii_file[0]):
            raise TypeError("Ending bounds out of Boundary")
        if bounds[1] <= bounds[0]:
            raise TypeError("Non-Valid boundary")

    # TODO: generalize this to more than the four cardinal directions
    direction_offsets = list(enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]))

    if not mario_version:
        tile_catalog, tile_grid, _code_list, _unique_tiles = make_tile_catalog(image, tile_size)
        (
            pattern_catalog,
            pattern_weights,
            pattern_list,
            pattern_grid,
        ) = make_pattern_catalog_with_rotations(
            tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations
        )
    else:
        # Convert ascii_file to list
        ascii_list = [list(row) for row in ascii_file]
        for row in ascii_list:
            if '\n' in row:
                row.remove('\n')
        ascii_list = [[ord(tok) for tok in row] for row in ascii_list]

        # Limit ascii_list to bounds, if given
        if bounds:
            if fix_outer_bounds:
                fixation = pattern_width - 1
                starting_bound = max(bounds[0] - fixation, 0)
                ending_bound = min(bounds[1] + fixation, len(ascii_list[0]) - 1) + 1
                output_size[0] += 2 * fixation
                #print(fixation, starting_bound, ending_bound)
            else:
                starting_bound = bounds[0]
                ending_bound = bounds[1]

            ascii_list = [row[starting_bound:ending_bound] for row in ascii_list]

        tile_grid, _code_list, _unique_tiles = make_mario_catalog(ascii_list)
        (
            pattern_catalog,
            pattern_weights,
            pattern_list,
            pattern_grid,
        ) = make_pattern_catalog_with_rotations(
            tile_grid, pattern_width, input_is_periodic=input_periodic, rotations=rotations
        )

    logger.debug("pattern catalog")

    # visualize_tiles(unique_tiles, tile_catalog, tile_grid)
    # visualize_patterns(pattern_catalog, tile_catalog, pattern_weights, pattern_width)
    # figure_list_of_tiles(unique_tiles, tile_catalog, output_filename=f"visualization/tilelist_{filename}_{timecode}")
    # figure_false_color_tile_grid(tile_grid, output_filename=f"visualization/tile_falsecolor_{filename}_{timecode}")
    if visualize and filename:
        figure_pattern_catalog(
            pattern_catalog,
            tile_catalog,
            pattern_weights,
            pattern_width,
            output_filename=f"visualization/pattern_catalog_{filename}_{timecode}",
        )

    logger.debug("profiling adjacency relations")
    if False:
        import pprofile  # type: ignore
        profiler = pprofile.Profile()
        with profiler:
            adjacency_relations = adjacency_extraction(
                pattern_grid,
                pattern_catalog,
                direction_offsets,
                [pattern_width, pattern_width],
            )
        profiler.dump_stats(f"logs/profile_adj_{filename}_{timecode}.txt")
    else:
        adjacency_relations = adjacency_extraction(
            pattern_grid,
            pattern_catalog,
            direction_offsets,
            (pattern_width, pattern_width),
        )
    logger.debug("adjacency_relations")

    if visualize:
        figure_adjacencies(
            adjacency_relations,
            direction_offsets,
            tile_catalog,
            pattern_catalog,
            pattern_width,
            [tile_size, tile_size],
            output_filename=f"visualization/adjacency_{filename}_{timecode}_A",
        )
        # figure_adjacencies(adjacency_relations, direction_offsets, tile_catalog, pattern_catalog, pattern_width, [tile_size, tile_size], output_filename=f"visualization/adjacency_{filename}_{timecode}_B", render_b_first=True)

    logger.debug(f"output size: {output_size}\noutput periodic: {output_periodic}")
    number_of_patterns = len(pattern_weights)
    logger.debug(f"# patterns: {number_of_patterns}")
    decode_patterns = dict(enumerate(pattern_list))
    encode_patterns = {x: i for i, x in enumerate(pattern_list)}
    _encode_directions = {j: i for i, j in direction_offsets}

    adjacency_list: Dict[Tuple[int, int], List[Set[int]]] = {}
    for _, adjacency in direction_offsets:
        adjacency_list[adjacency] = [set() for _ in pattern_weights]
    # logger.debug(adjacency_list)
    for adjacency, pattern1, pattern2 in adjacency_relations:
        # logger.debug(adjacency)
        # logger.debug(decode_patterns[pattern1])
        adjacency_list[adjacency][encode_patterns[pattern1]].add(encode_patterns[pattern2])

    logger.debug(f"adjacency: {len(adjacency_list)}")

    time_adjacency = time.perf_counter()

    ground_list, sky_list = set_ground_sky(pattern_grid, encode_patterns, ground, sky)

    # Fixate outer bounds if set
    bound_list: Optional[NDArray[np.int64]] = None
    if fix_outer_bounds and bounds is not None:
        bound_list = np.vectorize(lambda x: encode_patterns[x])(
            pattern_grid[:, :fixation]
        )
        bound_list2 = np.vectorize(lambda x: encode_patterns[x])(
            pattern_grid[:, -fixation:]
        )
        bound_list = np.concatenate((bound_list, bound_list2), axis=1)

    import sys
    np.set_printoptions(threshold=sys.maxsize)

    # Add additional patterns from determined difficulty slices
    if enable_difficulty_suggestion:
        for af in difficulty_list:
            # Convert ascii_file to list
            al = [list(row) for row in af]
            for row in al:
                if '\n' in row:
                    row.remove('\n')
            al = [[ord(tok) for tok in row] for row in al]

            # Pattern extraction of new level
            tg, cl, ut = make_mario_catalog(al)
            (pc, pw, pl, pg) = make_pattern_catalog_with_rotations(
                tg, pattern_width, input_is_periodic=input_periodic, rotations=rotations
            )

            # Update pattern list with new patterns, preserving indices
            combined_array = np.concatenate((pattern_list, pl))
            unique_values, unique_indices = np.unique(combined_array, return_index=True)
            pattern_list = combined_array[np.sort(unique_indices)]

            # Update Ground and Sky Elements
            encode_patterns = {x: i for i, x in enumerate(pattern_list)}
            decode_patterns = dict(enumerate(pattern_list))
            gl, sl = set_ground_sky(pg, encode_patterns, ground, sky)
            ground_list = ground_list.union(gl)
            sky_list = sky_list.union(sl)

            # Add adjacencies of new patterns
            ar = adjacency_extraction(
                pg,
                pc,
                direction_offsets,
                (pattern_width, pattern_width),
            )
            adjacency_relations = adjacency_relations.union(ar)

            # Update pattern catalog and weights
            pattern_catalog.update(pc)
            for key, value in pw.items():
                if key in pattern_weights:
                    pattern_weights[key] = value + pattern_weights[key]
                else:
                    pattern_weights[key] = value

            number_of_patterns = len(pattern_weights)
            print(number_of_patterns)

    print(pattern_catalog)
    wave = makeWave(
        number_of_patterns, output_size[0], output_size[1], fixation, ground=ground_list, sky=sky_list, bound=bound_list
    )

    adjacency_matrix = makeAdj(adjacency_list, number_of_patterns)

    ### Heuristics ###
    encoded_weights: NDArray[np.float64] = np.zeros((number_of_patterns), dtype=np.float64)
    for w_id, w_val in pattern_weights.items():
        encoded_weights[encode_patterns[w_id]] = w_val
    choice_random_weighting: NDArray[np.float64] = np.random.random_sample(wave.shape[1:]) * 0.1

    pattern_heuristic: Callable[[NDArray[np.bool_], NDArray[np.bool_]], int] = lexicalPatternHeuristic
    if choice_heuristic == "rarest":
        pattern_heuristic = makeRarestPatternHeuristic(encoded_weights)
    if choice_heuristic == "weighted":
        pattern_heuristic = makeWeightedPatternHeuristic(encoded_weights)
    if choice_heuristic == "random":
        pattern_heuristic = makeRandomPatternHeuristic(encoded_weights)

    logger.debug(loc_heuristic)
    location_heuristic: Callable[[NDArray[np.bool_]], Tuple[int, int]] = lexicalLocationHeuristic
    if loc_heuristic == "anti-entropy":
        location_heuristic = makeAntiEntropyLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "entropy":
        location_heuristic = makeEntropyLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "random":
        location_heuristic = makeRandomLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "simple":
        location_heuristic = simpleLocationHeuristic
    if loc_heuristic == "spiral":
        location_heuristic = makeSpiralLocationHeuristic(choice_random_weighting)
    if loc_heuristic == "hilbert":
        location_heuristic = makeHilbertLocationHeuristic(choice_random_weighting)

    ### Visualization ###

    (
        visualize_choice,
        visualize_wave,
        visualize_backtracking,
        visualize_propagate,
        visualize_final,
        visualize_after,
    ) = (None, None, None, None, None, None)
    if filename and visualize:
        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = make_solver_visualizers(
            f"{filename}_{timecode}",
            wave,
            decode_patterns=decode_patterns,
            pattern_catalog=pattern_catalog,
            tile_catalog=tile_catalog,
            tile_size=[tile_size, tile_size],
        )
    if filename and logging:
        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())
    if filename and logging and visualize:
        vis = make_solver_visualizers(
            f"{filename}_{timecode}",
            wave,
            decode_patterns=decode_patterns,
            pattern_catalog=pattern_catalog,
            tile_catalog=tile_catalog,
            tile_size=[tile_size, tile_size],
        )
        log = make_solver_loggers(f"{filename}_{timecode}", input_stats.copy())

        def visfunc(idx: int):
            def vf(*args, **kwargs):
                if vis[idx]:
                    vis[idx](*args, **kwargs)
                if log[idx]:
                    return log[idx](*args, **kwargs)

            return vf

        (
            visualize_choice,
            visualize_wave,
            visualize_backtracking,
            visualize_propagate,
            visualize_final,
            visualize_after,
        ) = [visfunc(x) for x in range(len(vis))]

    ### Global Constraints ###
    active_global_constraint = lambda wave: True
    if global_constraint == "allpatterns":
        active_global_constraint = make_global_use_all_patterns()
    logger.debug(active_global_constraint)
    combined_constraints = [active_global_constraint]

    def combinedConstraints(wave: NDArray[np.bool_]) -> bool:
        return all(fn(wave) for fn in combined_constraints)

    ### Solving ###

    time_solve_start = None
    time_solve_end = None

    solution_tile_grid = None
    logger.debug("solving...")
    attempts = 0
    while attempts < attempt_limit:
        attempts += 1
        #print(attempts)
        time_solve_start = time.perf_counter()
        stats = {}
        # profiler = pprofile.Profile()
        # with profiler:
        # with PyCallGraph(output=GraphvizOutput(output_file=f"visualization/pycallgraph_{filename}_{timecode}.png")):
        try:
            solution = run(
                wave.copy(),
                adjacency_matrix,
                locationHeuristic=location_heuristic,
                patternHeuristic=pattern_heuristic,
                periodic=output_periodic,
                backtracking=backtracking,
                onChoice=visualize_choice,
                onBacktrack=visualize_backtracking,
                onObserve=visualize_wave,
                onPropagate=visualize_propagate,
                onFinal=visualize_final,
                checkFeasible=combinedConstraints,
            )
            if visualize_after:
                stats = visualize_after()
            # logger.debug(solution)
            # logger.debug(stats)
            solution_as_ids = np.vectorize(lambda x: decode_patterns[x])(solution)
            solution_tile_grid = pattern_grid_to_tiles(
                solution_as_ids, pattern_catalog
            )

            logger.debug("Solution:")
            # logger.debug(solution_tile_grid)
            if filename and not mario_version:
                render_tiles_to_output(
                    solution_tile_grid,
                    tile_catalog,
                    (tile_size, tile_size),
                    os.path.join(output_destination, filename + "_" + timecode + ".png"),
                )
            if filename and mario_version:
                save_ascii_solution(solution_tile_grid,
                    os.path.join(output_destination, filename + "_" + timecode + ".txt"),
                )

            time_solve_end = time.perf_counter()
            stats.update({"outcome": "success"})
        except StopEarly:
            logger.debug("Skipping...")
            stats.update({"outcome": "skipped"})
            raise
        except TimedOut:
            logger.debug("Timed Out")
            if visualize_after:
                stats = visualize_after()
            stats.update({"outcome": "timed_out"})
        except Contradiction as exc:
            logger.warning(f"Contradiction: {exc}")
            if visualize_after:
                stats = visualize_after()
            stats.update({"outcome": "contradiction"})
        finally:
            # profiler.dump_stats(f"logs/profile_{filename}_{timecode}.txt")
            outstats = {}
            outstats.update(input_stats)
            solve_duration = time.perf_counter() - time_solve_start
            if time_solve_end is not None:
                solve_duration = time_solve_end - time_solve_start
            adjacency_duration = time_solve_start - time_adjacency
            outstats.update(
                {
                    "attempts": attempts,
                    "time_start": time_begin,
                    "time_adjacency": time_adjacency,
                    "adjacency_duration": adjacency_duration,
                    "time solve start": time_solve_start,
                    "time solve end": time_solve_end,
                    "solve duration": solve_duration,
                    "pattern count": number_of_patterns,
                }
            )
            outstats.update(stats)
            if log_stats_to_output is not None:
                log_stats_to_output(outstats, output_destination + log_filename + ".tsv")
        if solution_tile_grid is not None and not mario_version:
            return tile_grid_to_image(solution_tile_grid, tile_catalog, (tile_size, tile_size))
        if solution_tile_grid is not None and mario_version:
            # np solution to correct oriented list
            sol = list(np.flip(np.rot90(solution_tile_grid, k=1, axes=(0, 1)), axis=0))
            # convert ascii value to symbols
            sol = [[chr(tok) for tok in row] for row in sol]
            # conver ascii list to strings
            sol = ["".join(row) for row in sol]
            if fix_outer_bounds and bounds is not None:
                sol = [row[fixation:-fixation] for row in sol]
            #print("Finish")
            return sol

    raise TimedOut("Attempt limit exceeded.")
