from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

p1:Person={'name':'shubham','age':135}

print(p1)