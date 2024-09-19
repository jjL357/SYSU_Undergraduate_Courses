def isOK(Pages,M,read_page):#检测每个学生读read_page页书能否按照题目要求读完
    count_page = read_page#当前读书的学生还能读的页数
    students = 1#读完以及还在读书的学生数量
    for i in range(0,len(Pages)):
        page = Pages[i]
        if page <= count_page:#当前学生能读完这本书
            count_page -= page

        else:#当前学生读不完这本书，换下一个学生读
            count_page = read_page
            count_page -= page
            students += 1
    if students<=M:#学生人数能在不超过M的情况下读完
        return True
    return False
               
def find_min_page(Pages,M):#计算最小页数
    if len(Pages)==0 :#没有书可读 返回-
        return 0
    if M==0:#没有学生
        print("No solution!")
        return -1
    #min_page = max(Pages),max_page = sum(Pages)
    min_page,max_page=Pages[0],0
    for i in range(len(Pages)):
        min_page = max(min_page,Pages[i])
        max_page += Pages[i]

    #二分查找能满足要求的最小页数
    left , right = min_page,max_page
    while left<right:#停止循环条件
        mid = (left+right)//2
        #print(left,right,mid,isOK(Pages,M,mid))
        if isOK(Pages,M,mid):#当前容量能读完 搜索空间为[left,mid]
            right = mid
        else :#当前容量不能读完 搜索空间为[mid+1,right]
            left = mid + 1
    return right
    

def main():
    #处理输入数据
    Pages = []
    input_NM = input("请输入书的本数N和学生数M:").strip().split(' ')
    N,M = int(input_NM[0]),int(input_NM[1])
    print("请按顺序排列输入每本书的页数:")
    input_page = input().strip().split()
    for i in range(N):
        Pages .append(int(input_page[i]))
    #输出答案
    print("最小页数: ",find_min_page(Pages,M))
    
if __name__ == "__main__":
    main()

    