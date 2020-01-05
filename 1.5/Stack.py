#我们选用List的末端（index=-1）作为栈顶
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self,item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

mystack = Stack()
mystack.push(5)
mystack.push("考试90+")
mystack.push("完蛋")
mystack.push("丢绩点")
mystack.pop()
mystack.push("考试90+")
print(mystack.peek())

