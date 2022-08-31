import json

with open('./data/data.json','r') as pro:
    json_data = json.load(pro)
# print(json_data)
data = []
ddata = ('text',)
for i in range(len(json_data)):
    dddd = {}
    dddd['text'] = json_data[i]['text']
    data.append(dddd)
    # data.append((*ddata,json_data[i]['text']))

# print(data_process)
with open('./data/test1.json','w') as p:
    json.dump(data,p)
# print(json_data[0]['text'])