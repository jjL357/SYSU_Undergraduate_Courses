nm=input()
temp=nm.split(' ')
n=int(temp[0])
m=int(temp[1])
graph={}
letter_to_num={}
num_to_letter={}
cur_index=0
for i in range(m) :
    edge = input()
    if edge[0] in graph :
        graph[edge[0]].append([edge[2], int(edge[4])])
    else :
        graph[edge[0]] = [[edge[2], int(edge[4])]]
        letter_to_num[edge[0]]=cur_index
        num_to_letter[cur_index]=edge[0]
        cur_index+=1
    if edge[2] in graph :
        graph[edge[2]].append([edge[0], int(edge[4])])
    else :
        graph[edge[2]] = [[edge[0], int(edge[4])]]
        letter_to_num[edge[2]]=cur_index
        num_to_letter[cur_index]=edge[2]
        cur_index+=1

distance=[[float("inf")]*n for i in range(n)]
visited=[[0]*n for i in range(n)]
path=[['']*n for i in range(n)]

for i in range(n):
    distance[i][i]=0
    path[i][i]=num_to_letter[i]

for i in range(0,n):
    for j in range(0,n):
        min=float("inf")
        min_index=-1
        for k in range(0,n):
            if not visited[i][k] and min>distance[i][k]:
                min=distance[i][k]
                min_index=k
        visited[i][min_index]=1
        start=num_to_letter[i]
        mid=num_to_letter[min_index]
        for edge in graph[mid]:
           end=edge[0]
           value=edge[1]
           k=letter_to_num[end]
           if distance[i][k]>distance[i][min_index]+value:
                distance[i][k]=distance[i][min_index]+value
                path[i][k]=mid

while True:
    inq=input()
    if inq[0]=='-':break
    temp=inq.split()
    start=letter_to_num[temp[0]]
    end=letter_to_num[temp[1]]
    shortest_path=[]
    print(distance[start][end],end=' ')
    end=temp[1]
    while not end==temp[0]:
        shortest_path.append(end)
        end=path[start][letter_to_num[end]]
        shortest_path.append("->")
    shortest_path.append(temp[0])
    shortest_path.reverse()
    for i in range(0,len(shortest_path)): 
        print(shortest_path[i],end='')
    print()    

    