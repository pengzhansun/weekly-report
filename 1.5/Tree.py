class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

# T = BinaryTree(100)
#
# T.insertLeft(50)
# T.insertRight(40)
#
# T.preorder()

class BinHeap:

    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self,i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i//2]:
                tmp = self.heapList[i//2]
                self.heapList[i//2] = self.heapList[i]
                self.heapList[i] = tmp
            # 沿路径向上
            i = i // 2


    def insert(self,k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self,i):
        while (i*2) <= self.currentSize:
            mc = self.minChild(i)



    def minChild(self,i):
        if 2 * i + 1 > self.currentSize:
            return 2 * i#说明唯一子节点
        else:#返回较小者
            if self.heapList[i * 2] < self.heapList[2 * i + 1]:
                return 2 * i
            else:
                return 2 * i + 1


    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()#把列表里最后一个元素移除
        self.percDown(1)#新顶下沉
        return retval


    def buildHeap(self,alist):
        # 从最后节点的父节点开始
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        print(len(self.heapList), i)
        while i > 0:
            print(self.heapList, i)
            self.percDown(i)
            i = i - 1
        print(self.heapList,i)


