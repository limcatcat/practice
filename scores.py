kor, mat, eng, sci = map(int, input().split())


print(kor >= 90 and mat > 85 and eng > 80 and sci >= 80)
print((kor+mat+eng+sci)/4)
