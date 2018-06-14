import pdb
BLACK = 0
RED = 1

class LLNode(object):

    def __init__(self, start, end):
        self.__start = start
        self.__end = end
        self.__left = None
        self.__right = None

    def __repr__(self):
        return "(%d, %d)" % (self.__start, self.__end)

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, val):
        self.__left = val

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, val):
        self.__right = val

class Interval(object):

    def __init__(self, start, end):
        self.__start = start
        self.__end = end
        self.__max = end
        self.__left = None
        self.__right = None
        # for balancing
        self.__parent = None
        self.__color = BLACK

    def __repr__(self):
        return "[%s, %s]" % (self.start, self.end)

    @property
    def start(self):
        return self.__start

    @property
    def end(self):
        return self.__end

    @property
    def max(self):
        return self.__max

    @max.setter
    def max(self, val):
        self.__max = val

    @property
    def left(self):
        return self.__left

    def __update_max(self):
        c = self.max
        r = -1 if self.right is None else self.right.max
        l = -1 if self.left is None else self.left.max
        self.max = max(c, l, r)

    @left.setter
    def left(self, val):
        self.__left = val
        val.parent = self
        if self.max < val.max:
            self.__max = val.max
        self.__update_max()

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, val):
        self.__right = val
        val.parent = self
        if self.max < val.max:
            self.__max = val.max
        self.__update_max()

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, val):
        self.__parent = val

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, val):
        self.__color = val

    def compare(self, other):
        if self.start < other.start:
            return -1
        elif self.start == other.start:
            if self.end < other.end:
                return -1
            elif self.end == other.end:
                return 0
            else:
                return -1
        else:
            return 1

LEAF = Interval(-1,-1)

class IntervalTree(object):

    def __init__(self):
        self.__root = None

    @property
    def root(self):
        return self.__root

    def insert_old(self, new_node, tmp=None):
        if tmp is None:
            tmp == self.__root
        if new_node.end > tmp.max:
            tmp.max = new_node.end
        res = tmp.compare(new_node)
        if res <= 0:
            if tmp.right is None:
                tmp.right = new_node
            else:
                self.insert(new_node, self.right)
        else:
            if tmp.left is None:
                tmp.left = new_node
            else:
                self.insert(new_node, self.left)
        return new_node

    @classmethod
    def grandparent(cls, node):
        if node.parent is None:
            return None
        return node.parent

    @classmethod
    def sibling(cls, node):
        p = node.parent
        if p is None:
            return None
        if p.left == node:
            return p.right
        else:
            return p.left

    @classmethod
    def uncle(cls, node):
        p = node.parent
        g = cls.grandparent(node)
        if g is None:
            return None
        return cls.sibling(p)

    @classmethod
    def swap_child(cls, parent, child, new_child):
        if parent.left == child:
            parent.left == new_child
        else:
            parent.right = new_child

    @classmethod
    def rotate_left(cls, node):
        old_parent = node.parent
        nnew = node.right
        assert nnew != LEAF
        node.right = nnew.left
        nnew.left = node
        cls.swap_child(old_parent, node, nnew)

    @classmethod
    def rotate_right(cls, node):
        old_parent = node.parent
        nnew = node.left
        assert nnew != LEAF
        node.left = nnew.right
        nnew.right = node
        cls.swap_child(old_parent, node, nnew)

    @classmethod
    def insert_recurse(cls, root, n):
        if root is not None:
            if n.end > root.max:
                root.max = n.end
            res = root.compare(n)
            if res > 0:
                if root.left is not LEAF:
                    cls.insert_recurse(root.left, n)
                else:
                    root.left = n
            elif root is not None:
                if root.right is not LEAF:
                    cls.insert_recurse(root.right, n)
                else:
                    root.right = n
        n.parent = root
        n.left = LEAF
        n.right = LEAF
        n.color = RED

    @classmethod
    def insert_repair(cls, n):
        p = n.parent
        if p is None:
            n.color = BLACK
        elif p.color == BLACK:
            pass
        else:
            g = p.parent
            u = cls.uncle(n)
            if u.color == RED:
                p.color = BLACK
                u.color = BLACK
                g.color = RED
                cls.insert_repair(g)
            else:
                if n is g.left.right:
                    cls.rotate_left(n)
                    n = n.left
                elif n is g.right.left:
                    cls.rotate_right(n)
                    n = n.right
                p = n.parent
                g = p.parent
                if n == p.left:
                    cls.rotate_left(n)
                else:
                    cls.rotate_right(n)
                p.color = BLACK
                g.color = RED

    def insert(self, start, end):
        n = Interval(start, end)
        self.insert_recurse(self.root, n)
        self.insert_repair(n)
        self.__root = n
        p = self.__root.parent
        while p is not None:
            self.__root = p
            p = self.__root.parent
        return self.__root

    def query(self, point):
        ret = list()
        visits =  self.query_recurse(point, self.root, ret)
        return (ret, visits)

    def query_recurse(self, point, root, ret):
        print("comparing (%s, %s)" % (root.start, root.end))
        visits = 1
        if point <= root.end and point >= root.start:
            ret.append(root)
        if root.left is not None and root.left.max >= point:
            visits += self.query_recurse(point, root.left, ret)
        if root.right is not None:
            visits += self.query_recurse(point, root.right, ret)
        return visits

    def __repr__(self):
        l = list()
        self.walk(self.root, l)
        return ", ".join(repr(i) for i in l)

    def walk(self, root, l):
        if root.left is not None:
            self.walk(root.left, l)
        if root != LEAF:
            l.append(root)
        if root.right is not None:
            self.walk(root.right, l)


def find_intervals(p, intervals):
    ret = list()
    for i in intervals:
        if p >= i[0] and p <= i[1]:
            ret.append(i)
    return ret

def find_intervals_ll(p, head_node):
    ret = list()
    curr = head_node
    while curr:
        if p >= curr.start and p <= curr.end:
            ret.append(curr)
        curr = curr.right
    return ret

it = IntervalTree()

from random import randrange, seed
from time import time

seed(100)

print("Generating intervals")
dat = list()
head = None
nodes = list()
curr = None


for i in range(10):
    b1 = randrange(1000)
    b2 = randrange(1000)
    b1 = (i+1)*100
    b2 = (i+3)*100
    mn = min(b1, b2)
    mx = max(b1, b2)
    i = [mn, mx]
    dat.append(i)
    ll = LLNode(mn, mx)
    nodes.append(ll)
    if curr:
        curr.right = ll
        ll.left = curr
        curr = ll
    else:
        head = ll
        curr = ll

print(dat)

print("Building interval tree")
for i in dat:
    it.insert(i[0], i[1])

def compare(intervals, lists):
    if len(intervals) != len(lists):
        return "not the same length"
    for (i, (x,y)) in enumerate(zip(intervals, lists)):
        if x[0] != y[0] or x[1] != y[1]:
            return "%d: %s != %s" % (i, x, y)
    return 1


def benchmark(query):
    q = query
    t1 = time()
    (it_res, visits) = it.query(q)
    print(visits)
    t2 = time()
    it_time = t2-t1
    t1 = t2
    #bf_res = find_intervals(q, dat)
    bf_res = find_intervals_ll(q, head)
    t2 = time()
    bf_time = t2-t1
    it_res = sorted(map(lambda x: (x.start, x.end), it_res))
    #bf_res = sorted(map(lambda x: (x[0], x[1]), bf_res))
    bf_res = sorted(map(lambda x: (x.start, x.end), bf_res))
    comp_res = compare(it_res, bf_res)
    if comp_res != 1:
        raise Exception("unequal: %s" % comp_res)
    return it_time - bf_time

#tdi = list()
#for q in range(0, 1000, 5):
#    tdi.append(benchmark(q))
#
#import numpy as np
#print(np.mean(tdi), np.min(tdi), np.max(tdi))

print(it.query(300))
