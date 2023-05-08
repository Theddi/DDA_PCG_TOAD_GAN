package mff.agents.gridSearch;

public class GridSearchNode implements Comparable<GridSearchNode> {
    public int depth;
    public float cost;
    public GridSearchNode parent;

    public int tileX;
    public int tileY;

    public GridJumpDirection jumpDirection = GridJumpDirection.UNDEFINED;
    public GridJumpState jumpState = GridJumpState.ON_GROUND;
    public int jumpUpTravelled = 0;
    public int horizontalJumpBoostLeft = 0;

    public GridSearchNode(int tileX, int tileY, int depth, GridSearchNode parent) {
        this.tileX = tileX;
        this.tileY = tileY;
        this.depth = depth;
        this.parent = parent;
    }

    @Override
    public int compareTo(GridSearchNode other) {
        return Float.compare(this.cost, other.cost);
    }
}
