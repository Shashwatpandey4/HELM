from helm.backend.mesh import DeviceMesh
import pytest

def test_device_mesh_basic():
    # DP=2, PP=2, TP=2 (Total 8)
    mesh = DeviceMesh(dp=2, pp=2, tp=2)
    
    # Coordinates
    # Rank 0 -> (0,0,0) [DP=0, PP=0, TP=0]
    assert mesh.get_coordinate(0) == (0,0,0)
    
    # Rank 1 -> (0,0,1) [DP=0, PP=0, TP=1]
    # Rank 2 -> (0,1,0) [DP=0, PP=1, TP=0]
    assert mesh.get_coordinate(2) == (0,1,0)
    
    # Rank 4 -> (1,0,0) [DP=1, PP=0, TP=0]
    assert mesh.get_coordinate(4) == (1,0,0)
    
    # Reverse Lookup
    assert mesh.get_global_rank(0,0,0) == 0
    assert mesh.get_global_rank(0,1,0) == 2
    assert mesh.get_global_rank(1,0,0) == 4
    
def test_tp_groups():
    # DP=2, PP=1, TP=2 (Total 4)
    # Ranks: [0, 1] (Replica 0), [2, 3] (Replica 1)
    mesh = DeviceMesh(dp=2, pp=1, tp=2)
    
    # Rank 0 is in TP group with Rank 1 (Same Replica, Same Stage)
    # Expected: [0, 1]
    group0 = mesh.get_tp_group_ranks(0)
    assert 0 in group0 and 1 in group0 and len(group0) == 2
    
    # Rank 2 is in TP group with Rank 3
    group2 = mesh.get_tp_group_ranks(2)
    assert 2 in group2 and 3 in group2 and len(group2) == 2

def test_dp_groups():
    # DP=2, PP=1, TP=2
    # Rank 0 (Rep 0, TP 0) -> Is DP Peer with Rank 2 (Rep 1, TP 0)
    mesh = DeviceMesh(dp=2, pp=1, tp=2)
    
    group_dp = mesh.get_dp_group_ranks(0)
    # Expect [0, 2]
    expected = [0, 2]
    # Check if elements match (order might vary)
    assert sorted(group_dp) == sorted(expected)

if __name__ == "__main__":
    test_device_mesh_basic()
    test_tp_groups()
    test_dp_groups()
