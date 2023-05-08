package mff.agents.gridSearch;

// max jump height: 4 blocks
// max horizontal jump + start needed:
//      11 (14 tiles start)
//      10 (4 tiles start)
//       9 (2 tiles start)
//       7 (1 tile start)

// diagonal jumps, 2 tile start:
//  +0: 9 (size of the gap)
//  +1: 8
//  +2: 7
//  +3: 6
//  +4: 5

// jump through platforms
//  pass through from below, even through more of them
//  can't go through from the top
//  can pass from sides and walk in them

import engine.core.MarioEvent;
import engine.core.MarioWorld;
import engine.helper.TileFeature;
import mff.LevelLoader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.PriorityQueue;

import static engine.helper.TileFeature.*;

// TODO: check big Mario
// TODO: allow only JUMP action in my A* (and no action to allow free fall?)
// TODO: for now, let's use "2 tile start physics" everywhere
public class GridSearch {
    private static final int MAX_JUMP_HEIGHT = 4;
    private int horizontalJumpBoost = 0;
    private final PriorityQueue<GridSearchNode> opened = new PriorityQueue<>();
    private final HashMap<Integer, Float> visitedStates = new HashMap<>();

    // [0,0] at top left
    private final int[][] levelTiles;

    public int totalNodesVisited = 0;
    public boolean success = false;

    public GridSearch(int[][] levelTiles, int marioTileX, int marioTileY) {
        this.levelTiles = levelTiles;

        // Mario not placed in level layout and original framework did not find a proper start location
        if (marioTileY == -1)
            marioTileY = 0;

        opened.add(new GridSearchNode(marioTileX, marioTileY, 0, null));
    }

    public GridSearch(int[][] levelTiles, int marioTileX, int marioTileY, int horizontalJumpBoost) {
        this(levelTiles, marioTileX, marioTileY);
        this.horizontalJumpBoost = horizontalJumpBoost;
        assert opened.peek() != null;
        opened.peek().horizontalJumpBoostLeft = this.horizontalJumpBoost;
    }

    public static GridSearch initGridSearch(String levelPath) {
        String level = LevelLoader.getLevel(levelPath);
        MarioEvent[] killEvents = new MarioEvent[0];
        MarioWorld world = new MarioWorld(killEvents);
        world.initializeLevel(level, 1000000);
        int[][] levelTiles = world.level.getLevelTiles();
        int marioTileX = world.level.marioTileX;
        int marioTileY = world.level.marioTileY;
        return new GridSearch(levelTiles, marioTileX, marioTileY);
    }

    public static GridSearch initGridSearch(String levelPath, int horizontalJumpBoost) {
        String level = LevelLoader.getLevel(levelPath);
        MarioEvent[] killEvents = new MarioEvent[0];
        MarioWorld world = new MarioWorld(killEvents);
        world.initializeLevel(level, 1000000);
        int[][] levelTiles = world.level.getLevelTiles();
        int marioTileX = world.level.marioTileX;
        int marioTileY = world.level.marioTileY;
        return new GridSearch(levelTiles, marioTileX, marioTileY, horizontalJumpBoost);
    }

    public int[][] markGridPathInLevelTiles(ArrayList<GridSearchNode> path) {
        int[][] result = new int[this.levelTiles.length][this.levelTiles[0].length];
        for (GridSearchNode node : path) {
            result[node.tileX][node.tileY] = 1;
        }
        return result;
    }

    public ArrayList<GridSearchNode> findGridPath() {
        GridSearchNode furthest = opened.peek();
        while (opened.size() > 0) {
            GridSearchNode current = opened.remove();
            totalNodesVisited++;

            if (current.tileX > furthest.tileX)
                furthest = current;

            if (finished(current)) {
                success = true;
                return recoverFullPath(current);
            }

            ArrayList<GridMove> possibleMoves = getPossibleMoves(current);

            for (GridMove move : possibleMoves) {
                GridSearchNode newState = advance(current, move);

                float newStateCost = calculateCost(newState);

                int newStateCode = getStateCode(newState);
                float newStateOldCost = visitedStates.getOrDefault(newStateCode, -1f);
                if (newStateOldCost >= 0 && newStateOldCost <= newStateCost)
                    continue;

                visitedStates.put(newStateCode, newStateCost);
                newState.cost = newStateCost;
                opened.add(newState);
            }
        }

        return recoverFullPath(furthest);
//        throw new IllegalStateException("Level is not solvable!");
    }

    private float calculateCost(GridSearchNode newState) {
        return newState.depth + 1f * (levelTiles.length - newState.tileX) - newState.horizontalJumpBoostLeft;
    }

    private int getStateCode(GridSearchNode newState) {
        // int tileX = 17 bits (plenty)
        // int tileY (16, let's say 32 to be safe) = 6 bits
        // int jumpDirection (0-2) = 2 bits
        // int jumpState (0-10) = 4 bits
        // int jumpUpTravelled (0-4) = 3 bits
        int result = 0;
        result |= (newState.tileX << 15);
        result |= (newState.tileY << 9);
        result |= (newState.jumpDirection.ordinal() << 7);
        result |= (newState.jumpState.ordinal() << 3);
        result |= (newState.jumpUpTravelled);
        return result;
    }

    private boolean finished(GridSearchNode current) {
        return (levelTiles[current.tileX][current.tileY] == 39 ||
                levelTiles[current.tileX][current.tileY] == 40);
    }

    private GridSearchNode advance(GridSearchNode current, GridMove move) {
        // jump up on the same x coordinate (first right, then up, then back left),
        //   - move to the side and initiate jump if airborne with undefined jump direction
        //   - set state to WALKED_OF_AN_EDGE and handle in getPossibleMoves
        //     - we don't need to care too much, MFF A* can handle this easily

        int tileX = current.tileX;
        int tileY = current.tileY;
        switch (move) {
            case LEFT  -> tileX = current.tileX - 1;
            case RIGHT -> tileX = current.tileX + 1;
            case DOWN  -> tileY = current.tileY + 1;
            case UP   -> tileY = current.tileY - 1;
        }
        GridSearchNode next = new GridSearchNode(tileX, tileY, current.depth + 1, current);
        next.horizontalJumpBoostLeft = current.horizontalJumpBoostLeft;

        switch (current.jumpState) {
            case ON_GROUND -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // start falling/jumping
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.WALKED_OFF_AN_EDGE;
                            next.jumpUpTravelled = 0;
                        } else {
                            // move on the ground
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // start falling/jumping
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.WALKED_OFF_AN_EDGE;
                            next.jumpUpTravelled = 0;
                        } else {
                            // move on the ground
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        next.jumpDirection = GridJumpDirection.UNDEFINED;
                        next.jumpState = GridJumpState.STRAIGHT_UP;
                        next.jumpUpTravelled = 1;
                    }
                    case DOWN -> {
                        throw new IllegalStateException("Can't move down when on the ground.");
                    }
                }
            }
            case WALKED_OFF_AN_EDGE -> {
                switch (move) {
                    case RIGHT, LEFT -> {
                        // allow small jump over a 1 wide pit (special case with ceiling just above us)
                        next.jumpDirection = GridJumpDirection.UNDEFINED;
                        next.jumpState = GridJumpState.ON_GROUND;
                        next.jumpUpTravelled = 0;
                    }
                    case UP -> {
                        next.jumpDirection = GridJumpDirection.UNDEFINED;
                        next.jumpState = GridJumpState.STRAIGHT_UP;
                        next.jumpUpTravelled = 1;
                    }
                    case DOWN -> {
                        if (!isBelowFree(current)) {
                            // invalid move from current
                            throw new IllegalStateException("There must be a free tile under Mario when in WALKED_OFF_AN_EDGE.");
                        } else if (!isBelowFree(next)) {
                            // land (just walked off one block lower)
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            // falling
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
            case STRAIGHT_UP -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.UP_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = current.jumpUpTravelled;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.UP_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = current.jumpUpTravelled;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        next.jumpDirection = GridJumpDirection.UNDEFINED;
                        next.jumpUpTravelled = current.jumpUpTravelled + 1;
                        if (next.jumpUpTravelled < MAX_JUMP_HEIGHT)
                            next.jumpState = GridJumpState.STRAIGHT_UP;
                        else
                            next.jumpState = GridJumpState.TOP;
                    }
                    case DOWN -> {
                        throw new IllegalStateException("Can't move down when jumping straight up.");
                    }
                }
            }
            case UP_HORIZONTAL_LAST_MOVE_UP -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.UP_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = current.jumpUpTravelled;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.UP_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = current.jumpUpTravelled;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't move up again in UP_HORIZONTAL_LAST_MOVE_UP.");
                    }
                    case DOWN -> {
                        throw new IllegalStateException("Can't move down in UP_HORIZONTAL_LAST_MOVE_UP.");
                    }
                }
            }
            case UP_HORIZONTAL_LAST_MOVE_HORIZONTAL -> {
                switch (move) {
                    case RIGHT, LEFT -> {
                        // longer horizontal jump
                        next.jumpDirection = current.jumpDirection;
                        if (current.horizontalJumpBoostLeft == 0) {
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL_FINAL;
                        } else {
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL;
                            next.horizontalJumpBoostLeft = current.horizontalJumpBoostLeft - 1;
                        }
                        next.jumpUpTravelled = 0;
                    }
                    case UP -> {
                        next.jumpDirection = current.jumpDirection;
                        next.jumpUpTravelled = current.jumpUpTravelled + 1;
                        if (next.jumpUpTravelled < MAX_JUMP_HEIGHT)
                            next.jumpState = GridJumpState.UP_HORIZONTAL_LAST_MOVE_UP;
                        else
                            next.jumpState = GridJumpState.TOP;
                    }
                    case DOWN -> {
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpUpTravelled = 0;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_DOWN;
                        }
                    }
                }
            }
            case TOP -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from TOP state.");
                    }
                    case DOWN -> {
                        throw new IllegalStateException("Can't go down in TOP state.");
                    }
                }
            }
            case TOP_MOVED_HORIZONTAL -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next) && current.horizontalJumpBoostLeft == 0) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL_FINAL;
                            next.jumpUpTravelled = 0;
                        } else if (isBelowFree(next) && current.horizontalJumpBoostLeft > 0) {
                            // horizontal boost
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                            next.horizontalJumpBoostLeft = current.horizontalJumpBoostLeft - 1;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next) && current.horizontalJumpBoostLeft == 0) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL_FINAL;
                            next.jumpUpTravelled = 0;
                        } else if (isBelowFree(next) && current.horizontalJumpBoostLeft > 0) {
                            // horizontal boost
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.TOP_MOVED_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                            next.horizontalJumpBoostLeft = current.horizontalJumpBoostLeft - 1;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from TOP_MOVED_HORIZONTAL state.");
                    }
                    case DOWN -> {
                        next.horizontalJumpBoostLeft = this.horizontalJumpBoost;
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
            case TOP_MOVED_HORIZONTAL_FINAL -> {
                switch (move) {
                    case RIGHT -> {
                        throw new IllegalStateException("Can't move right from TOP_MOVED_HORIZONTAL_TWICE state.");
                    }
                    case LEFT -> {
                        throw new IllegalStateException("Can't move left from TOP_MOVED_HORIZONTAL_TWICE state.");
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from TOP_MOVED_HORIZONTAL_TWICE state.");
                    }
                    case DOWN -> {
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
            case STRAIGHT_DOWN -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from STRAIGHT_DOWN state.");
                    }
                    case DOWN -> {
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.STRAIGHT_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
            case DOWN_HORIZONTAL_LAST_MOVE_DOWN -> {
                switch (move) {
                    case RIGHT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.RIGHT;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case LEFT -> {
                        if (isBelowFree(next)) {
                            // continue the jump
                            next.jumpDirection = GridJumpDirection.LEFT;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL;
                            next.jumpUpTravelled = 0;
                        } else {
                            // land
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        }
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from DOWN_HORIZONTAL_LAST_MOVE_DOWN state.");
                    }
                    case DOWN -> {
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.STRAIGHT_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
            case DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL -> {
                switch (move) {
                    case RIGHT -> {
                        throw new IllegalStateException("Can't move right from DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL state.");
                    }
                    case LEFT -> {
                        throw new IllegalStateException("Can't move left from DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL state.");
                    }
                    case UP -> {
                        throw new IllegalStateException("Can't jump up from DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL state.");
                    }
                    case DOWN -> {
                        if (!isBelowFree(next)) {
                            next.jumpDirection = GridJumpDirection.UNDEFINED;
                            next.jumpState = GridJumpState.ON_GROUND;
                            next.jumpUpTravelled = 0;
                        } else {
                            next.jumpDirection = current.jumpDirection;
                            next.jumpState = GridJumpState.DOWN_HORIZONTAL_LAST_MOVE_DOWN;
                            next.jumpUpTravelled = 0;
                        }
                    }
                }
            }
        }

        if (next.jumpState == GridJumpState.ON_GROUND)
            next.horizontalJumpBoostLeft = this.horizontalJumpBoost;
        return next;
    }

    private ArrayList<GridMove> getPossibleMoves(GridSearchNode current) {
        // if falling, can only move in the direction in which the jump was initiated, or straight down
        // jump direction can be undefined - just falling, init of jumping on top of a block in the same column

        // died in a pit
        if (current.tileY == 15)
            return new ArrayList<>();

        var possibleMoves = new ArrayList<GridMove>();

        switch (current.jumpState) {
            case ON_GROUND -> {
                // right
                if (isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                // left
                if (isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
                // up
                if (isAboveFree(current))
                    possibleMoves.add(GridMove.UP);
            }
            case WALKED_OFF_AN_EDGE -> {
                // consider it a start of a jump
                if (isAboveFree(current))
                    possibleMoves.add(GridMove.UP);
                // consider it a start of a fall
                if (isBelowFree(current))
                    possibleMoves.add(GridMove.DOWN);
                // allow small jump over a 1 wide pit (special case with ceiling just above us)
                if (isRightFree(current) && !isFree(current.tileX + 1, current.tileY + 1, GridMove.DOWN))
                    possibleMoves.add(GridMove.RIGHT);
                if (isLeftFree(current) && !isFree(current.tileX - 1, current.tileY + 1, GridMove.DOWN))
                    possibleMoves.add(GridMove.LEFT);
            }
            case STRAIGHT_UP -> {
                // continue straight up
                if (isAboveFree(current) && current.jumpUpTravelled < MAX_JUMP_HEIGHT)
                    possibleMoves.add(GridMove.UP);
                // move horizontally
                if (current.jumpDirection == GridJumpDirection.RIGHT && isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (current.jumpDirection == GridJumpDirection.LEFT && isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
                // jump up on the same column
                if (current.jumpDirection == GridJumpDirection.UNDEFINED) {
                    if (isRightFree(current))
                        possibleMoves.add(GridMove.RIGHT);
                    if (isLeftFree(current))
                        possibleMoves.add(GridMove.LEFT);
                }
            }
            case TOP -> {
                // move horizontally
                if (current.jumpDirection == GridJumpDirection.RIGHT && isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (current.jumpDirection == GridJumpDirection.LEFT && isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
                // jump up on the same column
                if (current.jumpDirection == GridJumpDirection.UNDEFINED) {
                    if (isRightFree(current))
                        possibleMoves.add(GridMove.RIGHT);
                    if (isLeftFree(current))
                        possibleMoves.add(GridMove.LEFT);
                }
            }
            case UP_HORIZONTAL_LAST_MOVE_UP -> {
                // move horizontally
                if (current.jumpDirection == GridJumpDirection.RIGHT && isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (current.jumpDirection == GridJumpDirection.LEFT && isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
            }
            case UP_HORIZONTAL_LAST_MOVE_HORIZONTAL -> {
                // continue straight up
                if (isAboveFree(current) && current.jumpUpTravelled < MAX_JUMP_HEIGHT)
                    possibleMoves.add(GridMove.UP);
                // start falling down
                if (isBelowFree(current))
                    possibleMoves.add(GridMove.DOWN);
                // longer horizontal jump, go to state TOP_MOVED_HORIZONTAL_TWICE
                if (current.jumpDirection == GridJumpDirection.RIGHT && isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (current.jumpDirection == GridJumpDirection.LEFT && isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
            }
            case TOP_MOVED_HORIZONTAL, DOWN_HORIZONTAL_LAST_MOVE_DOWN -> {
                // move horizontally
                if (current.jumpDirection == GridJumpDirection.RIGHT && isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (current.jumpDirection == GridJumpDirection.LEFT && isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
                // start falling (straight) down
                if (isBelowFree(current))
                    possibleMoves.add(GridMove.DOWN);
            }
            case TOP_MOVED_HORIZONTAL_FINAL, DOWN_HORIZONTAL_LAST_MOVE_HORIZONTAL -> {
                // start or continue falling down
                if (isBelowFree(current))
                    possibleMoves.add(GridMove.DOWN);
            }
            case STRAIGHT_DOWN -> {
                // continue falling straight down
                if (isBelowFree(current))
                    possibleMoves.add(GridMove.DOWN);
                // add horizontal movement to the fall
                if (isRightFree(current))
                    possibleMoves.add(GridMove.RIGHT);
                if (isLeftFree(current))
                    possibleMoves.add(GridMove.LEFT);
            }
        }
        return  possibleMoves;
    }

    private boolean isAboveFree(GridSearchNode current) {
        return isFree(current.tileX, current.tileY - 1, GridMove.UP);
    }

    private boolean isLeftFree(GridSearchNode current) {
        return isFree(current.tileX - 1, current.tileY, GridMove.LEFT);
    }

    private boolean isRightFree(GridSearchNode current) {
        return isFree(current.tileX + 1, current.tileY, GridMove.RIGHT);
    }

    private boolean isBelowFree(GridSearchNode current) {
        return isFree(current.tileX, current.tileY + 1, GridMove.DOWN);
    }

    private boolean isFree(int tileX, int tileY, GridMove move) {
        // out of level bounds check
        if (tileX < levelTiles.length && tileX >= 0 &&
                tileY < levelTiles[0].length && tileY >= 0) {
            // BLOCK_UPPER = blocks from below (invisible blocks)
            // BLOCK_LOWER = blocks from above (pass through platforms)
            // BLOCK_ALL   = solid blocks
            // 48          = invisible one up block
            // 49          = invisible coin block
            ArrayList<TileFeature> tileFeatures = getTileType(levelTiles[tileX][tileY]);
            int tileValue = levelTiles[tileX][tileY];

            switch (move) {
                case LEFT, RIGHT -> {
                    return !tileFeatures.contains(TileFeature.BLOCK_ALL) &&
                            tileValue != 48 && tileValue != 49;
                }
                case DOWN -> {
                    return !(tileFeatures.contains(BLOCK_LOWER) || tileFeatures.contains(BLOCK_ALL)) &&
                            tileValue != 48 && tileValue != 49;
                }
                case UP -> {
                    return !(tileFeatures.contains(BLOCK_UPPER) || tileFeatures.contains(BLOCK_ALL)) &&
                            tileValue != 48 && tileValue != 49;
                }
                default -> throw new IllegalStateException("GridMove has forbidden value.");
            }
        } else {
            return false;
        }
    }

    private ArrayList<GridSearchNode> recoverFullPath(GridSearchNode finish) {
        var resultPath = new ArrayList<GridSearchNode>();
        GridSearchNode current = finish;
        while (current != null) {
            resultPath.add(current);
            current = current.parent;
        }
        Collections.reverse(resultPath);
        return resultPath;
    }
}
