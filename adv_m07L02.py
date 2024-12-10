# for n in range(2, 10):
#     for x in range (2, n):
#         if n % x == 0:
#             print(F"{n} equals {n} * {n//x}")
#             break
# print("------")
# for num in range (2,10):
#     if num % 2 == 0:
#         print(f"I found an even numer {num}")
#         continue
#     print(f"I found an odd number {num}")
print("------")
for n in range(2, 10):
    for x in range (2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        print(n, 'is a prime number')
print("------")
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')