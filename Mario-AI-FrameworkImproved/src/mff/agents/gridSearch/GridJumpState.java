package mff.agents.gridSearch;

/**
 * Represents the state that we reached after a move of a jump.
 */
public enum GridJumpState {
    ON_GROUND,
    WALKED_OFF_AN_EDGE,
    STRAIGHT_UP,
    UP_HORIZONTAL_LAST_MOVE_UP,
    UP_HORIZONTAL_LAST_MOVE_HORIZONTAL,
    TOP,
    TOP_MOVED_HORIZONTAL,
    TOP_MOVED_HORIZONTAL_FINAL,
    STRAIGHT_DOWN,
    DOWN_HORIZONTAL_LAST_MOVE_DOWN,
    DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL
}
