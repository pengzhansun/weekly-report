class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getdata(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, NewData):
        self.data = NewData

    def setNext(self,newnext):
        self.next = newnext

class UnorderedList:

    def __init__(self):
        self.head = None

    def add(self,item):
        temp = Node(item)
        temp.setNext(self.head)#temp的next赋值成为head
        self.head = temp#list的head赋值称为temp

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count + 1
            current = current.getNext()
        return count

    def search(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getdata() == item:
                found = True
            else:
                current = current.getNext()

            return found

    def remove(self, item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getdata() == item:
                found = True
            else:
                previous = current
                current - current.getNext()

        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())


mylist = UnorderedList()
mylist.add("你好")
mylist.add("此次考试为55分")
mylist.remove("此次考试为55分")
mylist.add("此次考试为95分")
print(mylist.search("此次考试为55分"))#False
