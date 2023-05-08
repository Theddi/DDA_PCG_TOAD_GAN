package mff.agents.astarGrid;

import mff.agents.astarHelper.CompareByCost;
import mff.agents.astarHelper.Helper;
import mff.agents.astarHelper.MarioAction;
import mff.agents.astarHelper.SearchNode;
import mff.agents.common.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;

public class AStarTree {
    private int[][] levelTilesWithPath;

    public SearchNode furthestNode;
    public float furthestNodeDistance;

    float marioXStart;
    int searchSteps;

    static boolean winFound = false;
    static final float maxMarioSpeedX = 10.91f;
    static float exitTileX;

    public int nodesEvaluated = 0;
    public int mostBacktrackedNodes = 0;
    private int farthestReachedX;
    private int nodesBeforeNewFarthestX = 0;

    public static float NODE_DEPTH_WEIGHT = 1f;
    public static float TIME_TO_FINISH_WEIGHT = 1.1f;
    public static float DISTANCE_FROM_PATH_TOLERANCE = 5;
    public static float DISTANCE_FROM_PATH_ADDITIVE_PENALTY = 0;
    public static float DISTANCE_FROM_PATH_MULTIPLICATIVE_PENALTY = 10;

    PriorityQueue<SearchNode> opened = new PriorityQueue<>(new CompareByCost());
    /**
     * INT STATE -> STATE COST
     */
    HashMap<Integer, Float> visitedStates = new HashMap<>();

    public AStarTree(MarioForwardModelSlim startState, int searchSteps, int[][] levelTilesWithPath) {
    	this.searchSteps = searchSteps;
        this.levelTilesWithPath = levelTilesWithPath;

    	marioXStart = startState.getMarioX();

    	furthestNode = getStartNode(startState);
    	furthestNode.cost = calculateCost(startState, furthestNode.nodeDepth);
    	furthestNodeDistance = furthestNode.state.getMarioX();

        farthestReachedX = (int) furthestNode.state.getMarioX();
    	
    	opened.add(furthestNode);
    }
    
    private int getIntState(MarioForwardModelSlim model) {
    	return getIntState((int) model.getMarioX(), (int) model.getMarioY());
    }
    
    private int getIntState(int x, int y) {
    	return (x << 16) | y;
    }
    
    private SearchNode getStartNode(MarioForwardModelSlim state) {
    	return new SearchNode(state);
    }
    
    private SearchNode getNewNode(MarioForwardModelSlim state, SearchNode parent, float cost, MarioAction action) {
    	return new SearchNode(state, parent, cost, action);
    }
    
    private float calculateCost(MarioForwardModelSlim nextState, int nodeDepth) {
        float timeToFinish = (exitTileX - nextState.getMarioX()) / maxMarioSpeedX;
        float distanceFromGridPathCost = calculateDistanceFromGridPathCost(nextState);
        return NODE_DEPTH_WEIGHT * nodeDepth
                + TIME_TO_FINISH_WEIGHT * timeToFinish
                + distanceFromGridPathCost;
	}

    private float calculateDistanceFromGridPathCost(MarioForwardModelSlim nextState) {
        int distanceFromGridPath = calculateDistanceFromGridPath(nextState);
        if (distanceFromGridPath <= DISTANCE_FROM_PATH_TOLERANCE)
            return 0;
        else
            return (distanceFromGridPath - DISTANCE_FROM_PATH_TOLERANCE)
                    * DISTANCE_FROM_PATH_MULTIPLICATIVE_PENALTY
                    + DISTANCE_FROM_PATH_ADDITIVE_PENALTY;
    }

    private int calculateDistanceFromGridPath(MarioForwardModelSlim nextState) {
        int marioTileX = (int) (nextState.getMarioX() / 16);
        int marioTileY = (int) (nextState.getMarioY() / 16);
        if (marioTileX >= 0 && marioTileX < levelTilesWithPath.length &&
            marioTileY >= 0 && marioTileY < levelTilesWithPath[0].length)
                if (levelTilesWithPath[marioTileX][marioTileY] == 1)
                    return 0;

        int distance = 1;
        do {
            // upper and lower side
            for (int x = marioTileX - distance; x <= marioTileX + distance; x++) {
                if (x < 0 || x >= levelTilesWithPath.length)
                    continue;

                int y = marioTileY - distance;
                if (y >= 0 && y < levelTilesWithPath[0].length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;

                y = marioTileY + distance;
                if (y >= 0 && y < levelTilesWithPath[0].length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;
            }
            // left and right side
            for (int y = marioTileY - distance + 1; y <= marioTileY + distance - 1; y++) {
                if (y < 0 || y >= levelTilesWithPath[0].length)
                    continue;

                int x = marioTileX - distance;
                if (x >= 0 && x < levelTilesWithPath.length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;

                x = marioTileX + distance;
                if (x >= 0 && x < levelTilesWithPath.length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;
            }
            distance++;
        } while (distance < 64);
        throw new IllegalStateException("Something seems wrong, distance to grid path shouldn't be this big.");
    }

    public ArrayList<boolean[]> search(MarioTimerSlim timer) {
        while (opened.size() > 0 && timer.getRemainingTime() > 0) {
            SearchNode current = opened.remove();
            nodesEvaluated++;

            if ((int) current.state.getMarioX() > farthestReachedX) {
                mostBacktrackedNodes = Math.max(nodesBeforeNewFarthestX, mostBacktrackedNodes);
                farthestReachedX = (int) current.state.getMarioX();
                nodesBeforeNewFarthestX = 0;
            } else {
                nodesBeforeNewFarthestX++;
            }

            if (current.state.getMarioX() > furthestNodeDistance) {
                furthestNode = current;
                furthestNodeDistance = current.state.getMarioX();
            }

            if (current.state.getGameStatusCode() == 1) {
                furthestNode = current;
                winFound = true;
                break;
            }

            ArrayList<MarioAction> actions = Helper.getPossibleActions(current.state);
            for (MarioAction action : actions) {
                MarioForwardModelSlim newState = current.state.clone();

                for (int i = 0; i < searchSteps; i++) {
                    newState.advance(action.value);
                }

                if (!newState.getWorld().mario.alive)
                    continue;

                float newStateCost = calculateCost(newState, current.nodeDepth + 1);

                int newStateCode = getIntState(newState);
                float newStateOldScore = visitedStates.getOrDefault(newStateCode, -1.0f);
                if (newStateOldScore >= 0 && newStateCost >= newStateOldScore)
                    continue;

                visitedStates.put(newStateCode, newStateCost);
                opened.add(getNewNode(newState, current, newStateCost, action));
            }
        }

        ArrayList<boolean[]> actionsList = new ArrayList<>();

        SearchNode curr = furthestNode;

        while (curr.parent != null) {
            for (int i = 0; i < searchSteps; i++) {
                actionsList.add(curr.marioAction.value);
            }
            curr = curr.parent;
        }

        return actionsList;
    }
}
