# import autograd.numpy as np
# from autograd.util import flatten_func
import numpy as np
from sklearn.utils import check_random_state
import random

# Batchization of the recursion

def batch(jets):
    # Batch the recursive activations across all nodes of a same level
    # !!! Assume that jets have at least one inner node.
    #     Leads to off-by-one errors otherwise :(

    # Reindex node IDs over all jets
    #
    # jet_children: array of shape [n_nodes, 2]
    #     jet_children[node_id, 0] is the node_id of the left child of node_id
    #     jet_children[node_id, 1] is the node_id of the right child of node_id
    #
    # jet_contents: array of shape [n_nodes, n_features]
    #     jet_contents[node_id] is the feature vector of node_id (4-vector in our case)
    
    jet_children = [] #List of trees, where each row is the tree of a jet
    offset = 0

    for jet in jets:
#         print('jet tree=',jet['tree'])
        tree = np.copy(jet["tree"])
        tree[tree != -1] += offset #We add an offset to every node that is not a leaf (SM)
        jet_children.append(tree) # We append the tree of each jet to jet_children. Each row corresponds to 1 jet.
        offset += len(tree) 

#     print('jet_children=',jet_children)
    jet_children = np.vstack(jet_children) #We concatenate all the jets tree  into 1 tree
#     print('jet_children=',jet_children)
    jet_contents = np.vstack([jet["content"] for jet in jets]) #We concatenate all the jet['contents'] into 1 array
#     print('jet_contents=',jet_contents)
    n_nodes = offset #Total number of nodes (we added the nodes of all the jets )
    #---------------------
    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32) #Array with 4 features per node
    level_children[:, [0, 2]] -= 1 #We set features 0 and 2 to -1. Features 0 and 2 will be the position of the left and right children of node_i, where node_i is given by "contents[node_i]" and left child is "content[level_children[node,0]]"

# 
#     # SANITY CHECK 1: Below we check that the jet_children contains the right location of each children subjet 
#     ii = -28
#     print('Content ',ii,' = ',jet_contents[ii])
#     print('Children location =',jet_children[ii])
#     if jet_children[ii][0]==-1: print('The node is a leaf')
#     else: print('Content ',ii,' by adding the 2 children 4-vectors= ',jet_contents[jet_children[ii,0]] 
#     + jet_contents[jet_children[ii,1]])

    inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category (SM)
    outers = []   # Outer nodes at level i       ---- The leaves are in this category (SM)
    offset = 0

    for jet in jets: # We fill the inners and outers array where each row corresponds to 1 level. We have each jet next to each other, so each jet root is a new column at depth 0, the first children add 2 columns at depth 1, .... Then we save in "level_children" the position of the left(right) child in the inners (or outers) array at depth i. So the 4-vector of node_i would be e.g. content[outers[level_children[i,0]]
    
        queue = [(jet["root_id"] + offset, -1, True, 0)] #(node, parent position, is_left, depth) 

        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0) #We pop the first element (This is expensive because we have to change the position of all the other tuples in the queue)

            if len(inners) < depth + 1:
                inners.append([]) #We append an empty list (1 per level) when the first node of a level shows up. 
            if len(outers) < depth + 1:
                outers.append([])

            # Inner node
            if jet_children[node, 0] != -1:#If node is not a leaf
                inners[depth].append(node) #We append the node to the inner list at row=depth because it has children
                position = len(inners[depth]) - 1 #position on the inners list of the last node we added
                is_leaf = False

                queue.append((jet_children[node, 0], node, True, depth + 1)) #Format: (node at the next level, parent node,"left", depth)
                queue.append((jet_children[node, 1], node, False, depth + 1))

            # Outer node
            else: #If the node is a leaf
                outers[depth].append(node)
                position = len(outers[depth]) - 1 #position on the outers list of the last node we added
                is_leaf = True

            # Register node at its parent. We save the position of the left and right children in the inners (or outers) array (at depth=depth_parent+1)
            if parent >= 0:
                if is_left:
                    level_children[parent, 0] = position #position of the left child in the inners (or outers) array (at depth=depth_parent+1)
                    level_children[parent, 1] = is_leaf #if True then the left child is a leaf => look in the outers array, else in the inners one
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"]) # We need this offset to get the right location in the jet_children array of each jet root node


# 
#     # SANITY CHECK 2: Below we check that the level_children contains the right location of each children subjet 
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=0
#     print('Root of jet #',ii+1,' location =',inners[level_parent][ii]) #The root is at level 0
#     print('Content jet #',ii+1,'=',jet_contents[inners[level_parent][ii]])
#     print('Children location:\n left=',inners[level_parent+1][level_children[inners[level_parent][ii],0]],'   right=',inners[level_parent+1][level_children[inners[level_parent][ii],2]])
#     if level_children[inners[level_parent][ii],1]==True: print('The node is a leaf')
#     else: print('Content ',inners[0][ii],' by adding the 2 children 4-vectors= ',jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],0]]] 
#     + jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],2]]])
# 
#     print('Is leaf at level ', level_parent,' = ', level_children[inners[level_parent][::],1])


    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []

    prev_inner = np.array([], dtype=int)

    for inner, outer in zip(inners, outers):
        n_inners.append(len(inner)) # We append the number of inner nodes in each level
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        levels.append(np.concatenate((inner, outer))) #Append the inners and outers of each level
        
 
        
        left = prev_inner[level_children[prev_inner, 1] == 1] # level_children[prev_inner, 1] returns a list with 1 for left children at level prev_inner+1 that are leaves and 0 otherwise. Then prev_inner[level_children[prev_inner, 1] == 1] picks the nodes at level prev_inner whose left children are leaves. So left are all nodes level prev_inner whose left child (at level prev_inner+1) is a leaf.
        level_children[left, 0] += len(inner) #We apply an offset to "left" because we concatenated inner and outer, with inners coming first. So now we get the right position of the children that are leaves in the levels array.
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)


        contents.append(jet_contents[levels[-1]]) # We append the 4-vector given by the nodes in the last row that we added to levels. This way we create a list of contents where each row corresponds to 1 level.
#         Then, the position of the left and right children in the levels list, will also be the position of them in the contents list, which is given by level_children Note that level_children keeps the old indices arrangement.

        prev_inner = inner #This will be the inner of the previous level in the next loop


#         print('level_children[prev_inner, 1] =',level_children[prev_inner, 1] )
#         print('left=',left)
#         print('right=',right)
#         print('prev_inner=',prev_inner)
#         print('contents=',contents)
#         print('length contents=',len(contents))
#         print('length levels =',len(levels))

# 
# #     # SANITY CHECK 3:
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=3
#     print('Final rearrangement of jets in batches')
#     print('Root of jet #',ii+1,' location =','level',level_parent,' pos:',ii) #The root is at level 0
#     print('Content jet #',ii+1,'=',contents[level_parent][ii])
#     print('Children location in the contents list','level',level_parent+1,'\n left=',level_children[levels[level_parent][ii],0],'   right=',level_children[levels[level_parent][ii],2])
#     if level_children[[ii],1]==True: print('The node is a leaf')
#     else: print('Content ','level',level_parent,' pos:',ii,' by adding the 2 children 4-vectors= ',contents[level_parent+1][level_children[levels[level_parent][ii],0]] 
#     + contents[level_parent+1][level_children[levels[level_parent][ii],2]])



    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 4]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 2] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_levels, n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]

    return (levels, level_children[:, [0, 2]], n_inners, contents)

# -----------------------------------------------------------------
# //////////////////////////////////////////////////////////////////

# Batchization of the recursion

def batch_Seb_no_pad(jets):
    # Batch the recursive activations across all nodes of a same level
    # !!! Assume that jets have at least one inner node.
    #     Leads to off-by-one errors otherwise :(

    # Reindex node IDs over all jets
    #
    # jet_children: array of shape [n_nodes, 2]
    #     jet_children[node_id, 0] is the node_id of the left child of node_id
    #     jet_children[node_id, 1] is the node_id of the right child of node_id
    #
    # jet_contents: array of shape [n_nodes, n_features]
    #     jet_contents[node_id] is the feature vector of node_id (4-vector in our case)
    
    jet_children =np.vstack([jet['tree'] for jet in jets])
#     print('jet_children=',jet_children)
#     jet_children = np.vstack(jet_children) #We concatenate all the jets tree  into 1 tree
#     print('jet_children=',jet_children)
    jet_contents = np.vstack([jet["content"] for jet in jets]) #We concatenate all the jet['contents'] into 1 array
#     print('jet_contents=',jet_contents)
    n_nodes=len(jet_children)

    #---------------------
    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32) #Array with 4 features per node
    level_children[:, [0, 2]] -= 1 #We set features 0 and 2 to -1. Features 0 and 2 will be the position of the left and right children of node_i, where node_i is given by "contents[node_i]" and left child is "content[level_children[node,0]]"

# 
#     # SANITY CHECK 1: Below we check that the jet_children contains the right location of each children subjet 
#     ii = -28
#     print('Content ',ii,' = ',jet_contents[ii])
#     print('Children location =',jet_children[ii])
#     if jet_children[ii][0]==-1: print('The node is a leaf')
#     else: print('Content ',ii,' by adding the 2 children 4-vectors= ',jet_contents[jet_children[ii,0]] 
#     + jet_contents[jet_children[ii,1]])

    inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category (SM)
    outers = []   # Outer nodes at level i       ---- The leaves are in this category (SM)
    offset = 0

    for jet in jets: # We fill the inners and outers array where each row corresponds to 1 level. We have each jet next to each other, so each jet root is a new column at depth 0, the first children add 2 columns at depth 1, .... Then we save in "level_children" the position of the left(right) child in the inners (or outers) array at depth i. So the 4-vector of node_i would be e.g. content[outers[level_children[i,0]]
    
        queue = [(jet["root_id"], -1, True, 0)] #(node, parent position, is_left, depth) 
        
        
        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0) #We pop the first element (This is expensive because we have to change the position of all the other tuples in the queue)

            if len(inners) < depth + 1:
                inners.append([]) #We append an empty list (1 per level) when the first node of a level shows up. 
            if len(outers) < depth + 1:
                outers.append([])

            # Inner node
            if jet_children[node, 0] != -1:#If node is not a leaf (it has a left child)
                inners[depth].append(node+offset) #We append the node to the inner list at row=depth because it has children
                position = len(inners[depth]) - 1 #position on the inners list of the last node we added
                is_leaf = False

                queue.append((jet_children[node+offset, 0], node+offset, True, depth + 1)) #Format: (node at the next level, parent node,"left", depth)
                queue.append((jet_children[node+offset, 1], node+offset, False, depth + 1))

            # Outer node
            else: #If the node is a leaf
                outers[depth].append(node+offset)
#                 print('outers=',outers)
                position = len(outers[depth]) - 1 #position on the outers list of the last node we added
                is_leaf = True

            # Register node at its parent. We save the position of the left and right children in the inners (or outers) array (at depth=depth_parent+1)
            if parent >= 0:
                if is_left:
                    level_children[parent, 0] = position #position of the left child in the inners (or outers) array (at depth=depth_parent+1)
                    level_children[parent, 1] = is_leaf #if True then the left child is a leaf => look in the outers array, else in the inners one
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"]) # We need this offset to get the right location in the jet_children array of each jet root node


# 
#     # SANITY CHECK 2: Below we check that the level_children contains the right location of each children subjet 
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=0
#     print('Root of jet #',ii+1,' location =',inners[level_parent][ii]) #The root is at level 0
#     print('Content jet #',ii+1,'=',jet_contents[inners[level_parent][ii]])
#     print('Children location:\n left=',inners[level_parent+1][level_children[inners[level_parent][ii],0]],'   right=',inners[level_parent+1][level_children[inners[level_parent][ii],2]])
#     if level_children[inners[level_parent][ii],1]==True: print('The node is a leaf')
#     else: print('Content ',inners[0][ii],' by adding the 2 children 4-vectors= ',jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],0]]] 
#     + jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],2]]])
# 
#     print('Is leaf at level ', level_parent,' = ', level_children[inners[level_parent][::],1])


    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []

    prev_inner = np.array([], dtype=int)
    print('----'*20)
    for inner, outer in zip(inners, outers):
        print('inner=',inner)
        print('outer=',outer)
        
        n_inners.append(len(inner)) # We append the number of inner nodes in each level
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        levels.append(np.concatenate((inner, outer))) #Append the inners and outers of each level
        
 
        
        left = prev_inner[level_children[prev_inner, 1] == 1] # level_children[prev_inner, 1] returns a list with 1 for left children at level prev_inner+1 that are leaves and 0 otherwise. Then prev_inner[level_children[prev_inner, 1] == 1] picks the nodes at level prev_inner whose left children are leaves. So left are all nodes level prev_inner whose left child (at level prev_inner+1) is a leaf.
        level_children[left, 0] += len(inner) #We apply an offset to "left" because we concatenated inner and outer, with inners coming first. So now we get the right position of the children that are leaves in the levels array.
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)


        contents.append(jet_contents[levels[-1]]) # We append the 4-vector given by the nodes in the last row that we added to levels. This way we create a list of contents where each row corresponds to 1 level.
#         Then, the position of the left and right children in the levels list, will also be the position of them in the contents list, which is given by level_children Note that level_children keeps the old indices arrangement.

        prev_inner = inner #This will be the inner of the previous level in the next loop


#         print('level_children[prev_inner, 1] =',level_children[prev_inner, 1] )
#         print('left=',left)
#         print('right=',right)
#         print('prev_inner=',prev_inner)
#         print('contents=',contents)
#         print('length contents=',len(contents))
#         print('length levels =',len(levels))

# 
# #     # SANITY CHECK 3:
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=3
#     print('Final rearrangement of jets in batches')
#     print('Root of jet #',ii+1,' location =','level',level_parent,' pos:',ii) #The root is at level 0
#     print('Content jet #',ii+1,'=',contents[level_parent][ii])
#     print('Children location in the contents list','level',level_parent+1,'\n left=',level_children[levels[level_parent][ii],0],'   right=',level_children[levels[level_parent][ii],2])
#     if level_children[[ii],1]==True: print('The node is a leaf')
#     else: print('Content ','level',level_parent,' pos:',ii,' by adding the 2 children 4-vectors= ',contents[level_parent+1][level_children[levels[level_parent][ii],0]] 
#     + contents[level_parent+1][level_children[levels[level_parent][ii],2]])



    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 4]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 2] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_levels, n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]
    
    print('n_inners[0]=',n_inners[0])
    return (levels, level_children[:, [0, 2]], n_inners, contents)

# //////////////////////////////////////////////////////////////////

# Batchization of the recursion

def batch_Seb(jets):
    # Batch the recursive activations across all nodes of a same level
    # !!! Assume that jets have at least one inner node.
    #     Leads to off-by-one errors otherwise :(

    # Reindex node IDs over all jets
    #
    # jet_children: array of shape [n_nodes, 2]
    #     jet_children[node_id, 0] is the node_id of the left child of node_id
    #     jet_children[node_id, 1] is the node_id of the right child of node_id
    #
    # jet_contents: array of shape [n_nodes, n_features]
    #     jet_contents[node_id] is the feature vector of node_id (4-vector in our case)
    
    jet_children =np.vstack([jet['tree'] for jet in jets])
#     print('jet_children=',jet_children)
#     jet_children = np.vstack(jet_children) #We concatenate all the jets tree  into 1 tree
#     print('jet_children=',jet_children)
    jet_contents = np.vstack([jet["content"] for jet in jets]) #We concatenate all the jet['contents'] into 1 array
#     print('jet_contents=',jet_contents)
    n_nodes=len(jet_children)

    #---------------------
    # Level-wise traversal
    level_children = np.zeros((n_nodes, 4), dtype=np.int32) #Array with 4 features per node
    level_children[:, [0, 2]] -= 1 #We set features 0 and 2 to -1. Features 0 and 2 will be the position of the left and right children of node_i, where node_i is given by "contents[node_i]" and left child is "content[level_children[node,0]]"

# 
#     # SANITY CHECK 1: Below we check that the jet_children contains the right location of each children subjet 
#     ii = -28
#     print('Content ',ii,' = ',jet_contents[ii])
#     print('Children location =',jet_children[ii])
#     if jet_children[ii][0]==-1: print('The node is a leaf')
#     else: print('Content ',ii,' by adding the 2 children 4-vectors= ',jet_contents[jet_children[ii,0]] 
#     + jet_contents[jet_children[ii,1]])

    inners = []   # Inner nodes at level i       ---- The nodes that are not leaves are in this category (SM)
    outers = []   # Outer nodes at level i       ---- The leaves are in this category (SM)
    offset = 0

    for jet in jets: # We fill the inners and outers array where each row corresponds to 1 level. We have each jet next to each other, so each jet root is a new column at depth 0, the first children add 2 columns at depth 1, .... Then we save in "level_children" the position of the left(right) child in the inners (or outers) array at depth i. So the 4-vector of node_i would be e.g. content[outers[level_children[i,0]]
    
        queue = [(jet["root_id"], -1, True, 0)] #(node, parent position, is_left, depth) 
        
        
        while len(queue) > 0:
            node, parent, is_left, depth = queue.pop(0) #We pop the first element (This is expensive because we have to change the position of all the other tuples in the queue)

            if len(inners) < depth + 1:
                inners.append([]) #We append an empty list (1 per level) when the first node of a level shows up. 
            if len(outers) < depth + 1:
                outers.append([])

            # Inner node
            if jet_children[node, 0] != -1:#If node is not a leaf (it has a left child)
                inners[depth].append(node+offset) #We append the node to the inner list at row=depth because it has children
                position = len(inners[depth]) - 1 #position on the inners list of the last node we added
                is_leaf = False

                queue.append((jet_children[node+offset, 0], node+offset, True, depth + 1)) #Format: (node at the next level, parent node,"left", depth)
                queue.append((jet_children[node+offset, 1], node+offset, False, depth + 1))

            # Outer node
            else: #If the node is a leaf
                outers[depth].append(node+offset)
#                 print('outers=',outers)
                position = len(outers[depth]) - 1 #position on the outers list of the last node we added
                is_leaf = True

            # Register node at its parent. We save the position of the left and right children in the inners (or outers) array (at depth=depth_parent+1)
            if parent >= 0:
                if is_left:
                    level_children[parent, 0] = position #position of the left child in the inners (or outers) array (at depth=depth_parent+1)
                    level_children[parent, 1] = is_leaf #if True then the left child is a leaf => look in the outers array, else in the inners one
                else:
                    level_children[parent, 2] = position
                    level_children[parent, 3] = is_leaf

        offset += len(jet["tree"]) # We need this offset to get the right location in the jet_children array of each jet root node


# 
#     # SANITY CHECK 2: Below we check that the level_children contains the right location of each children subjet 
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=0
#     print('Root of jet #',ii+1,' location =',inners[level_parent][ii]) #The root is at level 0
#     print('Content jet #',ii+1,'=',jet_contents[inners[level_parent][ii]])
#     print('Children location:\n left=',inners[level_parent+1][level_children[inners[level_parent][ii],0]],'   right=',inners[level_parent+1][level_children[inners[level_parent][ii],2]])
#     if level_children[inners[level_parent][ii],1]==True: print('The node is a leaf')
#     else: print('Content ',inners[0][ii],' by adding the 2 children 4-vectors= ',jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],0]]] 
#     + jet_contents[inners[level_parent+1][level_children[inners[level_parent][ii],2]]])
# 
#     print('Is leaf at level ', level_parent,' = ', level_children[inners[level_parent][::],1])


    # Reorganize levels[i] so that inner nodes appear first, then outer nodes
    levels = []
    n_inners = []
    contents = []
    n_level=[]

    prev_inner = np.array([], dtype=int)

    for inner, outer in zip(inners, outers):
        
        print('inner=',inner)
        print('outer=',outer)
        n_inners.append(len(inner)) # We append the number of inner nodes in each level
        inner = np.array(inner, dtype=int)
        outer = np.array(outer, dtype=int)
        levels.append(np.concatenate((inner, outer))) #Append the inners and outers of each level
        n_level.append(len(levels[-1]))
 
        
        left = prev_inner[level_children[prev_inner, 1] == 1] # level_children[prev_inner, 1] returns a list with 1 for left children at level prev_inner+1 that are leaves and 0 otherwise. Then prev_inner[level_children[prev_inner, 1] == 1] picks the nodes at level prev_inner whose left children are leaves. So left are all nodes level prev_inner whose left child (at level prev_inner+1) is a leaf.
        level_children[left, 0] += len(inner) #We apply an offset to "left" because we concatenated inner and outer, with inners coming first. So now we get the right position of the children that are leaves in the levels array.
        right = prev_inner[level_children[prev_inner, 3] == 1]
        level_children[right, 2] += len(inner)


        contents.append(jet_contents[levels[-1]]) # We append the 4-vector given by the nodes in the last row that we added to levels. This way we create a list of contents where each row corresponds to 1 level.
#         Then, the position of the left and right children in the levels list, will also be the position of them in the contents list, which is given by level_children Note that level_children keeps the old indices arrangement.

        prev_inner = inner #This will be the inner of the previous level in the next loop


#         print('level_children[prev_inner, 1] =',level_children[prev_inner, 1] )
#         print('left=',left)
#         print('right=',right)
#         print('prev_inner=',prev_inner)
#         print('contents=',contents)
#         print('length contents=',len(contents))
#         print('length levels =',len(levels))

# 
# #     # SANITY CHECK 3:
#     ii = 1 #location of the parent in the inner list at level_parent
#     level_parent=3
#     print('Final rearrangement of jets in batches')
#     print('Root of jet #',ii+1,' location =','level',level_parent,' pos:',ii) #The root is at level 0
#     print('Content jet #',ii+1,'=',contents[level_parent][ii])
#     print('Children location in the contents list','level',level_parent+1,'\n left=',level_children[levels[level_parent][ii],0],'   right=',level_children[levels[level_parent][ii],2])
#     if level_children[[ii],1]==True: print('The node is a leaf')
#     else: print('Content ','level',level_parent,' pos:',ii,' by adding the 2 children 4-vectors= ',contents[level_parent+1][level_children[levels[level_parent][ii],0]] 
#     + contents[level_parent+1][level_children[levels[level_parent][ii],2]])
    
    #We loop over the levels to zero pad the array (only a few levels per jet)
#     pad_levels=[]
#     pad_contents=[]
    n_inners=np.asarray(n_inners)
    max_n_level=np.max(n_level)
#     print('max_n_level=',max_n_level)
    print('----'*20)
    for i in range(len(levels)): 
#       print('max_n_level-len(levels[i])=',max_n_level-len(levels[i]))
      pad_dim=int(max_n_level-len(levels[i]))
#       levels[i]=np.pad(levels[i], (0,max_n_level-len(levels[i])), 'constant', constant_values=(0))
      levels[i]=np.concatenate((levels[i],np.zeros((pad_dim))))  
      contents[i]=np.concatenate((contents[i],np.zeros((pad_dim,4,1))))
#       contents[i]=np.pad(contents[i], (0,max_n_level-len(contents[i])), 'constant', constant_values=(0)) 
#       print(np.concatenate((contents[i],np.zeros((10,4,1)))))
#       pad_contents.append(np.concatenate((contents[i],np.zeros((pad_dim,4,1)))))  
#       pad_levels.append(element)
#     print('contents=',contents)
    
         
#       element=np.pad(element, (0,max_n_level-len(element)), 'constant', constant_values=(0))  
#       pad_levels.append(element)

    # levels: list of arrays
    #     levels[i][j] is a node id at a level i in one of the trees
    #     inner nodes are positioned within levels[i][:n_inners[i]], while
    #     leaves are positioned within levels[i][n_inners[i]:]
    #
    # level_children: array of shape [n_nodes, 4]
    #     level_children[node_id, 0] is the position j in the next level of
    #         the left child of node_id
    #     level_children[node_id, 2] is the position j in the next level of
    #         the right child of node_id
    #
    # n_inners: list of shape len(levels)
    #     n_inners[i] is the number of inner nodes at level i, accross all
    #     trees
    #
    # contents: array of shape [n_levels, n_nodes, n_features]
    #     contents[sum(len(l) for l in layers[:i]) + j] is the feature vector
    #     or node layers[i][j]

    return (levels, level_children[:, [0, 2]], n_inners, contents, n_level)
# -----------------------------------------------
# Split the sample into train, cross-validation and test
def split_sample(sig, bkg, train_frac_rel, val_frac_rel, test_frac_rel):
  
  print('---'*20)
  print('Loading trees ...')
  
  rndstate = random.getstate()
  random.seed(0)
  size=np.minimum(len(sig),len(bkg))
  print('sg length=',len(sig))
  
  sig_label=np.ones((size),dtype=int)
  bkg_label=np.zeros((size),dtype=int)
  
  train_frac=train_frac_rel
  val_frac=train_frac+val_frac_rel
  test_frac=val_frac+test_frac_rel

  N_train=int(train_frac*size)
  Nval=int(val_frac*size)
  Ntest=int(test_frac*size)
  #--------------------
  train_x=np.concatenate((sig[0:N_train],bkg[0:N_train]))
  train_y=np.concatenate((sig_label[0:N_train],bkg_label[0:N_train]))
  
  dev_x=np.concatenate((sig[N_train:Nval],bkg[N_train:Nval]))
  dev_y=np.concatenate((sig_label[N_train:Nval],bkg_label[N_train:Nval]))
  
  test_x=np.concatenate((sig[Nval:Ntest],bkg[Nval:Ntest]))
  test_y=np.concatenate((sig_label[Nval:Ntest],bkg_label[Nval:Ntest]))

  #---------------------
  indices_train = check_random_state(1).permutation(len(train_x))
  train_x = train_x[indices_train]
  train_y = train_y[indices_train]
  
  indices_dev = check_random_state(2).permutation(len(dev_x))
  dev_x = dev_x[indices_dev]
  dev_y = dev_y[indices_dev]
  
  indices_test = check_random_state(3).permutation(len(test_x))
  test_x = test_x[indices_test]
  test_y = test_y[indices_test]
  #----------------------
  
  train_x=np.asarray(train_x)
  dev_x=np.asarray(dev_x)
  test_x=np.asarray(test_x)
  train_y=np.asarray(train_y)
  dev_y=np.asarray(dev_y)
  test_y=np.asarray(test_y)  
  
  
  print('Train shape=',train_x.shape)
  #We reshape for single jets studies (Modify the code for full events)
#   train=train.reshape(train.shape[0]*train.shape[1])
#   dev=dev.reshape(dev.shape[0]*dev.shape[1])
# #   print(test.shape)
#   test=test.reshape(test.shape[0]*test.shape[1]) 
  print('Size data each sg and bg =',size)
  
  return train_x, train_y, dev_x, dev_y, test_x, test_y



def batch_array(sample_x,sample_y,batch_size):
  batches=[]
  for i in range(batch_size,len(sample_x)+1,batch_size): #This way even the last batch has the same size (We lose a few events at the end, with N_events<batch_size)
    batches.append([])
    levels, children, n_inners, contents, n_level= batch_Seb(sample_x[i-batch_size:i])
    batches[-1].append(levels)
    batches[-1].append(children)
    batches[-1].append(n_inners)
    batches[-1].append(contents)
    batches[-1].append(sample_y[i-batch_size:i])
    batches[-1].append(n_level)
    
  batches=np.asarray(batches)
  return batches
  
  
  
  
def batch_array_no_pad(sample_x,sample_y,batch_size):
  batches=[]
  for i in range(batch_size,len(sample_x)+1,batch_size): #This way even the last batch has the same size (We lose a few events at the end, with N_events<batch_size)
    batches.append([])
    levels, children, n_inners, contents= batch_Seb_no_pad(sample_x[i-batch_size:i])
    batches[-1].append(levels)
    batches[-1].append(children)
    batches[-1].append(n_inners)
    batches[-1].append(contents)
    
  batches=np.asarray(batches)
  return batches