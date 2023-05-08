package mff.agents.astarWaypoints;

import engine.helper.TileFeature;
import mff.agents.astarHelper.MarioAction;
import mff.agents.benchmark.IAgentBenchmark;
import mff.agents.benchmark.IAgentBenchmarkBacktrack;
import mff.agents.common.IGridHeuristic;
import mff.agents.common.IGridWaypoints;
import mff.agents.common.IMarioAgentMFF;
import mff.agents.common.MarioTimerSlim;
import mff.agents.gridSearch.GridSearchNode;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

import java.util.ArrayList;

import static engine.helper.TileFeature.BLOCK_ALL;
import static engine.helper.TileFeature.BLOCK_LOWER;

public class Agent implements IMarioAgentMFF, IAgentBenchmark, IGridHeuristic, IAgentBenchmarkBacktrack, IGridWaypoints {
    private ArrayList<boolean[]> actionsList = new ArrayList<>();
    private AStarTree tree;
    private boolean finished = false;
    private int totalSearchCalls = 0;

    @Override
    public void initialize(MarioForwardModelSlim model) {
        AStarTree.exitTileX = model.getWorld().level.exitTileX * 16;
        initializeWaypoints(AStarTree.gridPath, model);
        tree = new AStarTree();
    }

    private void initializeWaypoints(ArrayList<GridSearchNode> gridPath, MarioForwardModelSlim model) {
        int waypointsSpacing = AStarTree.WAYPOINT_DENSITY - 1;

        for (GridSearchNode node : gridPath) {
            waypointsSpacing++;

            if (waypointsSpacing < AStarTree.WAYPOINT_DENSITY)
                continue;

            byte blockValue = model.getWorld().level.getBlockValue(node.tileX, node.tileY + 1);
            ArrayList<TileFeature> tileFeatures = TileFeature.getTileType(blockValue);
            // only save waypoint if it's on the ground
            if (tileFeatures.contains(BLOCK_LOWER) || tileFeatures.contains(BLOCK_ALL)) {
                AStarTree.waypoints.add(new AStarTree.Waypoint(node.tileX * 16, node.tileY * 16));
                waypointsSpacing = 0;
            }
        }

        AStarTree.Waypoint lastIncludedWaypoint = AStarTree.waypoints.get(AStarTree.waypoints.size() - 1);
        GridSearchNode lastNodeOnPath = gridPath.get(gridPath.size() - 1);
        if (lastIncludedWaypoint.x == lastNodeOnPath.tileX && lastIncludedWaypoint.y == lastNodeOnPath.tileY)
            return;
        AStarTree.waypoints.add(new AStarTree.Waypoint(lastNodeOnPath.tileX * 16, lastNodeOnPath.tileY * 16));
    }

    @Override
    public void receiveLevelWithPath(int[][] levelTilesWithPath) {
        AStarTree.levelTilesWithPath = levelTilesWithPath;
    }

    @Override
    public void receiveGridPath(ArrayList<GridSearchNode> gridPath) {
        AStarTree.gridPath = gridPath;
    }

    @Override
    public boolean[] getActions(MarioForwardModelSlim model, MarioTimerSlim timer) {
        if (finished) {
            if (actionsList.size() == 0)
                return MarioAction.NO_ACTION.value;
            else
                return actionsList.remove(actionsList.size() - 1);
        }

        for (int i = 0; i < actionsList.size(); i++) {
            model.advance(actionsList.get(actionsList.size() - (1 + i)));
        }
        tree.initNewSearch(model);
        ArrayList<boolean[]> actionsFromFarthestPos = tree.search(timer);
        totalSearchCalls++;

        ArrayList<boolean[]> newActions = new ArrayList<>();
        newActions.addAll(actionsFromFarthestPos);
        newActions.addAll(actionsList);
        actionsList = newActions;

        if (tree.winFound) {
            finished = true;
            return actionsList.remove(actionsList.size() - 1);
        }

        if (actionsList.size() == 0) {
            return MarioAction.NO_ACTION.value;
        }

        return actionsList.remove(actionsList.size() - 1);
    }

    @Override
    public int getSearchCalls() {
        return totalSearchCalls;
    }

    @Override
    public int getNodesEvaluated() {
        return tree.nodesEvaluated;
    }

    @Override
    public int getMostBacktrackedNodes() {
        return tree.mostBacktrackedNodes;
    }

    @Override
    public String getAgentName() {
        return "MFF A* Waypoints Agent";
    }
}
