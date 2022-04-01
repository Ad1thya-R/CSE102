import math
import operator


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    def size(self):
        'Return size of binary tree'
        aout=0
        if self.right is None and self.left is None:
            return 1+aout
        if self.right is None:
            return 1+self.left.size()
        if self.left is None:
            return 1+self.right.size()
        else:
            return 1+self.right.size()+self.left.size()
def size(root):
        'Return size of binary tree'
        if root is None:
            return 0
        if root.right is None and root.left is None:
            return 1
        if root.right is None:
            return 1+size(root.left)
        if root.left is None:
            return 1+size(root.right)
        else:
            return 1+size(root.left)+size(root.right)
def sum_values(root):
    'return sum of values of binary tree'
    if root is None:
        return 0
    if root.right is None and root.left is None:
        return root.value
    if root.right is None:
        return root.value + sum_values(root.left)
    if root.left is None:
        return root.value + sum_values(root.right)
    else:
        return root.value + sum_values(root.right) + sum_values(root.left)

def height(root):
    'return the height of the binary tree'

    if root is None:
        return -1
    if root.right is None and root.left is None:
        return 0
    if root.right is None:
        return 1+height(root.left)
    if root.left is None:
        return 1+height(root.right)
    else:
        return 1+max(height(root.left),height(root.right))

def mirrored(lroot, rroot):
    'check whether 2 binary trees are mirrors of each other'
    if lroot is None and rroot is None:
        return True
    # If only one is empty
    if lroot is None or rroot is None:
        return False
    return lroot.value==rroot.value and mirrored(lroot.left,rroot.right) and mirrored(lroot.right,rroot.left)

def check_symmetry(root):
    if root is None:
        return True
    if root.left is None and root.right is None:
        return True
    # If only one is empty
    if root.left is None or root.right is None:
        return False
    return mirrored(root.left,root.right)

def accumulate(root, f,start):
    'return accumulation of values based on an operation f performed on binary tree'
    if root is None:
        return start
    if root.right is None and root.left is None:
        return f(start,root.value)
    if root.right is None:
        return f(start, accumulate(root.left,f,start))
    if root.left is None:
        return f(start, accumulate(root.right,f,start))
    else:
        return f(accumulate(root.right,f,start), accumulate(root.left,f,start))

INT_MAX = 4294967296
INT_MIN = -4294967296


def check_BST(root, floor=float('-inf'), ceiling=float('inf')):
    if not root:
        return True
    if root.value <= floor or root.value >= ceiling:
        return False
    # in the left branch, root is the new ceiling; contrarily root is the new floor in right branch
    return check_BST(root.left, floor, root.value) and check_BST(root.right, root.value, ceiling)



def min_BST(root):
    'return the minimum value in the binary search tree'
    if root is None:
        return math.inf
    current = root

    # loop down to find the leftmost leaf
    while (current.left is not None):
        current = current.left

    return current.value


T1 = Node(0, Node(1, Node(2), Node(3)), Node(4, Node(5)))
T2 = Node(0, Node(1), Node(2, Node(3), Node(4, Node(5))))
T3 = Node(11, Node(10, Node(9, None, Node(8)), Node(7)), Node(6))
print(size(T1),size(T2),size(T3))
print(sum_values(T1),sum_values(T2),sum_values(T3))
print(height(T1), height(T2), height(T3))
print(accumulate(T1, operator.add, 6))