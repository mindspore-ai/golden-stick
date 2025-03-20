# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test NameTree."""


import unittest
import os
import pytest
from mindspore_gs.common.name_prefix_tree import NameTree


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestNameTree(unittest.TestCase):
    """TestNameTree"""
    def setUp(self):
        self.tree = NameTree()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_add_name(self):
        """
        Feature: Adding names to NameTree.
        Description: Tests adding names to NameTree, including successful addition and duplicate addition scenarios.
        Expectation: No exception on successful addition, raises RuntimeError on duplicate addition.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)
        self.tree.add_name("x.y", 3)
        self.tree.add_name("x.z", 4)

        with self.assertRaises(RuntimeError):
            self.tree.add_name("a.b.c", 5)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes(self):
        """
        Feature: Getting sibling leaf nodes.
        Description: Tests getting sibling leaf nodes of a specified node, including normal cases, non-existent nodes,
                     and non-leaf nodes.
        Expectation: Returns the correct list of sibling leaf nodes, empty list for non-existent nodes, and raises
                     RuntimeError for non-leaf nodes.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)
        self.tree.add_name("a.b.e.f", 3)
        self.tree.add_name("x.y", 4)
        self.tree.add_name("x.z", 5)

        leafs = self.tree.get_sibling_leaf_nodes("a.b.c")
        assert len(leafs) == 2
        assert leafs[0].name == 'a.b.d'
        assert leafs[1].name == 'a.b.e.f'
        leafs = self.tree.get_sibling_leaf_nodes("x.y")
        assert len(leafs) == 1
        assert leafs[0].name == 'x.z'

        with self.assertRaises(RuntimeError):
            self.tree.get_sibling_leaf_nodes("a.b")

        self.assertEqual(self.tree.get_sibling_leaf_nodes("nonexistent.node"), [])

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes_normal_case(self):
        """
        Feature: Getting sibling leaf nodes (normal case).
        Description: Tests getting sibling leaf nodes at different levels (level=1, level=2, and level=3).
        Expectation: Returns the correct list of sibling leaf nodes with names matching the expectations.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)
        self.tree.add_name("a.e.f", 3)
        self.tree.add_name("a.e.g", 4)
        self.tree.add_name("h.i.j", 5)

        # Test with level=1
        nodes = self.tree.get_sibling_leaf_nodes("a.b.c", level=1)
        assert len(nodes) == 1
        assert nodes[0].name == "a.b.d"
        nodes = self.tree.get_sibling_leaf_nodes("a.b.d", level=1)
        assert len(nodes) == 1
        assert nodes[0].name == "a.b.c"
        nodes = self.tree.get_sibling_leaf_nodes("a.e.f", level=1)
        assert len(nodes) == 1
        assert nodes[0].name == "a.e.g"
        nodes = self.tree.get_sibling_leaf_nodes("a.e.g", level=1)
        assert len(nodes) == 1
        assert nodes[0].name == "a.e.f"
        assert self.tree.get_sibling_leaf_nodes("h.i.j", level=1) == []

        # Test with level=2
        nodes = self.tree.get_sibling_leaf_nodes("a.b.c", level=2)
        assert len(nodes) == 3
        assert nodes[0].name in ["a.e.f", "a.e.g", "a.b.d"]
        assert nodes[1].name in ["a.e.f", "a.e.g", "a.b.d"]
        assert nodes[2].name in ["a.e.f", "a.e.g", "a.b.d"]
        nodes = self.tree.get_sibling_leaf_nodes("a.b.d", level=2)
        assert len(nodes) == 3
        assert nodes[0].name in ["a.e.f", "a.e.g", "a.b.c"]
        assert nodes[1].name in ["a.e.f", "a.e.g", "a.b.c"]
        assert nodes[2].name in ["a.e.f", "a.e.g", "a.b.c"]
        nodes = self.tree.get_sibling_leaf_nodes("a.e.f", level=2)
        assert len(nodes) == 3
        assert nodes[0].name in ["a.b.c", "a.b.d", "a.e.g"]
        assert nodes[1].name in ["a.b.c", "a.b.d", "a.e.g"]
        assert nodes[2].name in ["a.b.c", "a.b.d", "a.e.g"]
        nodes = self.tree.get_sibling_leaf_nodes("a.e.g", level=2)
        assert len(nodes) == 3
        assert nodes[0].name in ["a.b.c", "a.b.d", "a.e.f"]
        assert nodes[1].name in ["a.b.c", "a.b.d", "a.e.f"]
        assert nodes[2].name in ["a.b.c", "a.b.d", "a.e.f"]
        assert self.tree.get_sibling_leaf_nodes("h.i.j", level=2) == []

        # Test with level=3
        nodes = self.tree.get_sibling_leaf_nodes("a.b.c", level=3)
        assert len(nodes) == 4
        assert nodes[0].name in ["a.e.f", "a.e.g", "a.b.d", "h.i.j"]
        assert nodes[1].name in ["a.e.f", "a.e.g", "a.b.d", "h.i.j"]
        assert nodes[2].name in ["a.e.f", "a.e.g", "a.b.d", "h.i.j"]
        assert nodes[3].name in ["a.e.f", "a.e.g", "a.b.d", "h.i.j"]

        nodes = self.tree.get_sibling_leaf_nodes("a.b.d", level=3)
        assert len(nodes) == 4
        assert nodes[0].name in ["a.e.f", "a.e.g", "a.b.c", "h.i.j"]
        assert nodes[1].name in ["a.e.f", "a.e.g", "a.b.c", "h.i.j"]
        assert nodes[2].name in ["a.e.f", "a.e.g", "a.b.c", "h.i.j"]
        assert nodes[3].name in ["a.e.f", "a.e.g", "a.b.c", "h.i.j"]

        nodes = self.tree.get_sibling_leaf_nodes("a.e.f", level=3)
        assert len(nodes) == 4
        assert nodes[0].name in ["a.b.c", "a.b.d", "a.e.g", "h.i.j"]
        assert nodes[1].name in ["a.b.c", "a.b.d", "a.e.g", "h.i.j"]
        assert nodes[2].name in ["a.b.c", "a.b.d", "a.e.g", "h.i.j"]
        assert nodes[3].name in ["a.b.c", "a.b.d", "a.e.g", "h.i.j"]

        nodes = self.tree.get_sibling_leaf_nodes("a.e.g", level=3)
        assert len(nodes) == 4
        assert nodes[0].name in ["a.b.c", "a.b.d", "a.e.f", "h.i.j"]
        assert nodes[1].name in ["a.b.c", "a.b.d", "a.e.f", "h.i.j"]
        assert nodes[2].name in ["a.b.c", "a.b.d", "a.e.f", "h.i.j"]
        assert nodes[3].name in ["a.b.c", "a.b.d", "a.e.f", "h.i.j"]

        nodes = self.tree.get_sibling_leaf_nodes("h.i.j", level=3)
        assert len(nodes) == 4
        assert nodes[0].name in ["a.b.c", "a.b.d", "a.e.f", "a.e.g"]
        assert nodes[1].name in ["a.b.c", "a.b.d", "a.e.f", "a.e.g"]
        assert nodes[2].name in ["a.b.c", "a.b.d", "a.e.f", "a.e.g"]
        assert nodes[3].name in ["a.b.c", "a.b.d", "a.e.f", "a.e.g"]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes_level_zero(self):
        """
        Feature: Getting sibling leaf nodes (level=0).
        Description: Tests getting sibling leaf nodes when the level parameter is 0.
        Expectation: Returns an empty list.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)

        assert self.tree.get_sibling_leaf_nodes("a.b.c", level=0) == []

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes_level_exceeds_tree_depth(self):
        """
        Feature: Getting sibling leaf nodes (level exceeds tree depth).
        Description: Tests getting sibling leaf nodes when the level parameter exceeds the tree depth.
        Expectation: Returns an empty list.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)

        assert self.tree.get_sibling_leaf_nodes("a.b.c", level=4) == []

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes_node_not_found(self):
        """
        Feature: Getting sibling leaf nodes (node not found).
        Description: Tests getting sibling leaf nodes of a non-existent node.
        Expectation: Returns an empty list.
        """
        self.tree.add_name("a.b.c", 1)

        assert self.tree.get_sibling_leaf_nodes("a.b.d", level=1) == []

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_sibling_leaf_nodes_non_leaf_node(self):
        """
        Feature: Getting sibling leaf nodes (non-leaf node).
        Description: Tests getting sibling leaf nodes of a non-leaf node.
        Expectation: Raises RuntimeError.
        """
        self.tree.add_name("a.b.c", 1)
        self.tree.add_name("a.b.d", 2)
        self.tree.add_name("a.b.e.f", 3)

        with pytest.raises(RuntimeError):
            self.tree.get_sibling_leaf_nodes("a.b", level=1)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_empty_tree(self):
        """
        Feature: Operations on an empty tree.
        Description: Tests getting sibling leaf nodes and finding leaf nodes with keywords in an empty tree.
        Expectation: Returns empty lists.
        """
        self.assertEqual(self.tree.get_sibling_leaf_nodes("root"), [])
        self.assertEqual(self.tree.find_leafs_with_keywords(["a"]), [])

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_single_node(self):
        """
        Feature: Operations on a single-node tree.
        Description: Tests getting sibling leaf nodes and finding leaf nodes with keywords in a single-node tree.
        Expectation: Returns empty list for sibling nodes and correct node for keyword search.
        """
        self.tree.add_name("single.node", 1)
        self.assertEqual(self.tree.get_sibling_leaf_nodes("single.node"), [])
        nodes = self.tree.find_leafs_with_keywords(["single"])
        assert len(nodes) == 1
        assert nodes[0].name == "single.node"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_generate_dot(self):
        """
        Feature: Generating DOT files.
        Description: Tests generating DOT files and cleaning up the generated files.
        Expectation: Successfully generates DOT files, files exist, and files are removed after testing.
        """
        self.tree.add_name("network.server1", 1)
        self.tree.add_name("network.server2", 2)
        self.tree.add_name("network.database.db1", 3)
        self.tree.add_name("network.database.db2", 4)
        self.tree.add_name("network.client.client1", 5)

        self.tree.generate_dot("output/tree.dot")
        self.assertTrue(os.path.exists("output/tree.dot"))

        self.tree.generate_dot("output/subdirectory/another_tree.dot")
        self.assertTrue(os.path.exists("output/subdirectory/another_tree.dot"))

        os.remove("output/tree.dot")
        os.remove("output/subdirectory/another_tree.dot")
        os.rmdir("output/subdirectory")
        os.rmdir("output")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_find_names_with_keywords(self):
        """
        Feature: Finding leaf nodes with keywords.
        Description: Tests finding leaf nodes containing keywords, including single and multiple keywords.
        Expectation: Returns the correct list of leaf nodes.
        """
        self.tree.add_name("network.server1", 1)
        self.tree.add_name("network.server2", 2)
        self.tree.add_name("network.database.db1", 3)
        self.tree.add_name("network.database.db2", 4)
        self.tree.add_name("network.client.client1", 5)

        result = self.tree.find_leafs_with_keywords(["server", "1"])
        self.assertEqual(result[0].name, "network.server1")

        results = sorted(self.tree.find_leafs_with_keywords(["database", "db"]))
        assert results[0].name == "network.database.db1"
        assert results[1].name == "network.database.db2"

        result = self.tree.find_leafs_with_keywords(["client", "1"])
        self.assertEqual(result[0].name, "network.client.client1")

        result = self.tree.find_leafs_with_keywords(["nonexistent"])
        self.assertEqual(result, [])

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_find_names_with_keywords_and_blacklist(self):
        """
        Feature: Finding leaf nodes with keywords and blacklist.
        Description: Tests finding leaf nodes containing keywords while excluding nodes in the blacklist.
        Expectation: Returns the correct list of leaf nodes, excluding those in the blacklist.
        """
        self.tree.add_name("network.server1", 1)
        self.tree.add_name("network.server2", 2)
        self.tree.add_name("network.database.db1", 3)
        self.tree.add_name("network.database.db2", 4)
        self.tree.add_name("network.client.client1", 5)

        result = self.tree.find_leafs_with_keywords(["server", "1"], blacklist=["network.server1"])
        self.assertEqual(result, [])

        result = self.tree.find_leafs_with_keywords(["database", "db"], blacklist=["network.database.db1"])
        self.assertEqual(result[0].name, "network.database.db2")

        result = self.tree.find_leafs_with_keywords(["client", "1"], blacklist=["network.client.client1"])
        self.assertEqual(result, [])

        result = self.tree.find_leafs_with_keywords(["nonexistent"])
        self.assertEqual(result, [])

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_find_leafs_with_prefix(self):
        """
        Feature: Finding leaf nodes with prefix.
        Description: Tests finding leaf nodes with a specified prefix, including existing and non-existent prefixes.
        Expectation: Returns the correct list of leaf nodes matching the prefix.
        """
        # Add some nodes
        self.tree.add_name("layer1.conv.weight", 1)
        self.tree.add_name("layer1.conv.bias", 2)
        self.tree.add_name("layer1.norm.weight", 3)
        self.tree.add_name("layer2.conv.weight", 4)
        self.tree.add_name("layer2.norm.bias", 5)
        self.tree.add_name("layer3.fc.weight", 6)

        # Test 1: Prefix exists and has multiple matching nodes
        prefix = "layer1.conv"
        result = sorted(self.tree.find_leafs_with_prefix(prefix))
        assert len(result) == 2
        assert result[0].name == "layer1.conv.bias"
        assert result[1].name == "layer1.conv.weight"
        print(f"Test 1 Passed")

        # Test 2: Prefix exists but has no matching nodes
        prefix = "layer1.linear"
        expected = []
        result = self.tree.find_leafs_with_prefix(prefix)
        assert result == expected, f"Test 2 Failed: Expected {expected}, got {result}"
        print(f"Test 2 Passed")

        # Test 3: Prefix is the root node
        prefix = "network"
        results = sorted(self.tree.find_leafs_with_prefix(prefix))
        assert len(results) == 6
        assert results[0].name == "layer1.conv.bias"
        assert results[1].name == "layer1.conv.weight"
        assert results[2].name == "layer1.norm.weight"
        assert results[3].name == "layer2.conv.weight"
        assert results[4].name == "layer2.norm.bias"
        assert results[5].name == "layer3.fc.weight"
        print(f"Test 3 Passed")

        # Test 4: Prefix is a leaf node
        prefix = "layer1.conv.weight"
        result = self.tree.find_leafs_with_prefix(prefix)
        assert result[0].name == prefix, f"Test 4 Failed: Expected {prefix}, got {result}"
        print(f"Test 4 Passed")

        # Test 5: Prefix does not exist
        prefix = "layer4"
        expected = []
        result = self.tree.find_leafs_with_prefix(prefix)
        assert result == expected, f"Test 5 Failed: Expected {expected}, got {result}"
        print(f"Test 5 Passed")

        # Test 6: Prefix is an intermediate node
        prefix = "layer2"
        expected = ["layer2.conv.weight", "layer2.norm.bias"]
        result = sorted(self.tree.find_leafs_with_prefix(prefix))
        assert len(result) == 2
        assert result[0].name == "layer2.conv.weight"
        assert result[1].name == "layer2.norm.bias"
        print(f"Test 6 Passed")

        print("All tests passed!")


if __name__ == "__main__":
    unittest.main()
