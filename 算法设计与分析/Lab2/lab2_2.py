def findRepeatedSubstr(str):#找出str中最长的重复出现并且非重叠的子字符串
    max_len,ans = 0, ""
    for i in range(len(str)):
        for j in range(i+1,len(str)):
            for l in range(1,len(str)-i-1):
                if j+l-1 >= len(str):#越界
                    continue
                if str[i:i+l] == str[j:j+l]:
                    if max_len < l:
                        max_len = l
                        ans = str[i:l]
    return ans

                           
def main():
    str = input("请输入字符串:")
    ans = findRepeatedSubstr(str)
    #输出结果
    print("最长且无重叠重复子字符串:",ans)
        
if __name__ == "__main__":
    main()
