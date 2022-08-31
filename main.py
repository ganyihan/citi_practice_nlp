# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
import json

json_obj = {'a':[1,2]}
json_obj1 = {"b": [9,4]}

json_obj["a"].extend(json_obj1['b'])
print(json_obj)