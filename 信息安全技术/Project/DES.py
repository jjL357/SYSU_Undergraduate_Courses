import numpy as np
class DES:
    # 初始化密钥和子密钥
    def __init__(self,key):
        self.key = self.hex2bin(key) # 密钥
        self.subkeys = self.generate_subkey() # 子密钥
    
    
    # 将十六进制表示的数字字符串转为二进制数保存在列表中(列表中每一位0或1)
    def hex2bin(self,input):
        # 将每个十六进制数转换为二进制表示，并存储在列表中
        binary_list = [bin(int(hex_char, 16))[2:].zfill(4) for hex_char in input]

        # 将二进制字符串转换为整数列表
        binary_list = [int(bit) for binary_string in binary_list for bit in binary_string]

        return binary_list
    
    # 将二进制数列表转为十六进制表示的数字字符串
    def bin2hex(self,binary_list):
        # 将二进制列表分割成每四位一组，并转换为对应的十六进制字符
        hexadecimal_number = ''
        for i in range(0, len(binary_list), 4):
            binary_char = ''.join([str(bit) for bit in binary_list[i:i+4]])
            hexadecimal_number += hex(int(binary_char, 2))[2:]

        return hexadecimal_number
    
    # 二进制数组转十进制整数
    def bin2dec(self,binary_list):
        dec = 0
        for bin in binary_list:
            dec *= 2
            dec += bin
        return dec
    
    # 十进制数转为二进制列表
    def int2bin(self,a, n):
        assert 0 <= n and a < 2**n
        res = [0] * n

        for x in range(n):
            res[n - x - 1] = a % 2
            a = a // 2
        return res
    
     # 异或操作(按数组中的每一位相对应异或)
    def xor(self,a,b):
        return [x^y for x, y in zip(a, b)]
    
    # 初始置换
    def initial_permutations(self,input):    
        ip = [58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7]  
        return [input[i-1] for i in ip]
    
    # 最终置换
    def final_permutations(self,input):
        fp = [40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25]
        return [input[i-1] for i in fp]

    
    
    # 循环左移offset位
    def leftRotate(self,a, offset):
        return a[offset:] + a[:offset]
    
    # PC2置换
    def PC2(self,key):
        pc2 = [14, 17, 11, 24, 1, 5, 
                3, 28, 15, 6, 21, 10,
                23, 19, 12, 4, 26, 8,
                16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]
        return [key[i-1] for i in pc2]
    
    # 根据密钥生成子密钥
    def generate_subkey(self):
         # PC1置换
        pc1_L = [57, 49, 41, 33, 25, 17, 9, 
            1, 58, 50, 42, 34, 26, 18, 
            10, 2, 59, 51, 43, 35, 27, 
            19, 11, 3, 60, 52, 44, 36]
        pc1_R = [63, 55, 47, 39, 31, 23, 15, 
            7, 62, 54, 46, 38, 30, 22, 
            14, 6, 61, 53, 45, 37, 29, 
            21, 13, 5, 28, 20, 12, 4]
        
        left = [self.key[i-1] for i in pc1_L]
        right = [self.key[i-1] for i in pc1_R]
        
        offset = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
        res = []

        for x in range(16):
            # LeftRotate
            left = self.leftRotate(left, offset[x])
            right = self.leftRotate(right, offset[x])
            # PC2置换产生子钥匙
            res.append(self.PC2(left + right))
        return res
    
    # 扩张置换，将32位的数据扩展到48位
    def Expand(self,input):
        e = [32, 1, 2, 3, 4, 5,
        4, 5, 6, 7, 8, 9,
        8, 9, 10, 11, 12, 13,
        12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21,
        20, 21, 22, 23, 24, 25,
        24, 25, 26, 27, 28, 29,
        28, 29, 30, 31, 32, 1]
        return [input[i-1] for i in e]
        

    
    # S盒变换，输入48位，输出32位
    def S(self,a):
        assert len(a) == 48

        S_box = [[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
                    0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
                    4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
                    15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13], 
                    [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
                    3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
                    0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
                    13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9], 
                    [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
                    13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
                    13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
                    1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12],
                    [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
                    13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
                    10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
                    3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14],
                    [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
                    14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
                    4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
                    11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3],
                    [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11,
                    10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
                    9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
                    4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13],
                    [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
                    13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
                    1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
                    6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12],
                    [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
                    1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
                    7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
                    2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]

        a = np.array(a, dtype=int).reshape(8, 6)
        res = []

        for i in range(8):
            # 用 S_box[i] 处理6位a[i]，得到4位输出
            p = a[i]
            r = S_box[i][self.bin2dec([p[0], p[5], p[1], p[2], p[3], p[4]])]
            res.append(self.int2bin(r, 4))
            
        res = np.array(res).flatten().tolist()
        assert len(res) == 32

        return res
    
    # P置换
    def P(self,a):
        p = [16, 7, 20, 21,
            29, 12, 28, 17,
            1, 15, 23, 26,
            5, 18, 31, 10,
            2, 8, 24, 14,
            32, 27, 3, 9,
            19, 13, 30, 6,
            22, 11, 4, 25]
        return [a[x-1] for x in p]
    
    # Feistel函数
    def F(self,RightHalf,subkey):
        t = self.xor(self.Expand(RightHalf), subkey)
        t = self.S(t)
        t = self.P(t)
        return t
    
    # 轮转
    def round(self,LeftHalf,RightHalf,subkeys): 
        for i in range(0,16):
            tmp = [x for x in RightHalf]
            F_result = self.F(RightHalf,subkeys[i])
            RightHalf = self.xor(LeftHalf,F_result)
            LeftHalf = tmp
        return  RightHalf + LeftHalf # 注意最后结果左右顺序调换
         
    #加密
    def encrypt(self,input):
        input = self.hex2bin(input)
        ip = self.initial_permutations(input)
        LeftHalf = ip[0:32]
        RightHalf = ip[32:]
        output = self.round(LeftHalf,RightHalf,self.subkeys)
        binary = self.final_permutations(output)
        return self.bin2hex(binary)
    
    # 解密
    def decrypt(self,input):
        input = self.hex2bin(input)
        ip = self.initial_permutations(input)
        LeftHalf = ip[0:32]
        RightHalf = ip[32:]
        output = self.round(LeftHalf,RightHalf,self.subkeys[::-1])
        binary = self.final_permutations(output)
        return self.bin2hex(binary)
    
   
    

    
    

    
    
    

# 测试代码
if __name__ == "__main__":
    
    # 测试数据(明文和密钥都为十六进制表示)
    plaintext = "edeabbccddeeffff"
    key = "cafababedeadbeaf"
    
    des = DES(key)
    
    
    #des = DES(key)
    
    
    # 加密
    ciphertext = des.encrypt(plaintext)
    print("加密后的密文:", ciphertext)
    
    # 解密
    decrypted_text = des.decrypt(ciphertext)
    print("解密后的明文:", decrypted_text)
