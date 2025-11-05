# import matplotlib.pyplot as plt
# import numpy as np

# def f(x):
#     f = x ** 2 + x - 4
#     return f

# def f2(x):
#     if (4 - x) < 0:
#         raise Exception("Cannot take square root of a negative number")
#     return (4 - x)**0.5

# def f_prime(x):
#     return 2 * x + 1

# def iteration(f, f2, n):
#     for i in range(1000):
#         if abs(f2(n) - n) < 1e-6:
#             return n
#         print(n)
#         n = f2(n)
#     raise Exception("didn't converge")


# def newton_raphson(f, f_prime, n):
#     for i in range(1000):
#         if abs(f(n)) < 1e-6:
#             return n
#         n = n - (f(n) / f_prime(n))
#     raise("didn't converge")

# def bisetion(f, a, b):
#     if f(a) * f(b) > 0:
#         raise("won't work")
#     c = (a + b) / 2
#     if(abs(f(c)) < 1e-11):
#         # print(c)
#         return c
#     elif f(a) * f(c) < 0:
#         b = c
#         return bisetion(f, a, b)
#     else:
#         a = c
#         return bisetion(f, a, b)

# def false_pos(f, a, b):
#     if f(a) * f(b) > 0:
#         raise("won't work")
#     c = b - f(b) * (a-b) / ((f(a) - f(b)))
#     if(abs(f(c)) < 1e-11):
#         # print(c)
#         return c
#     elif f(a) * f(c) < 0:
#         b = c
#         return bisetion(f, a, b)
#     else:
#         a = c
#         return bisetion(f, a, b)

# # sol = false_pos(f, 0, 10)
# # sol = newton_raphson(f, f_prime, 1)
# sol = iteration(f, f2, 1)
# print(sol)

# x = np.arange(-10, 10)
# print(x)

# plt.plot(x,f(x))
# plt.scatter(sol, 0)

# plt.grid()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x ** 2 - 2

# 1. Bisection

# def bisection(f, a , b, tol = 1e-6):
#     if(f(a) * f(b)) >= 0:
#         return "Error!"
#     while True:
#         c = (a + b) / 2
#         if(abs(f(c)) < tol):
#             return c
        
#         if(f(a) * f(c) < 0):
#             b = c
#         else:
#             a = c
# root = bisection(f, 1, 2)

# 2. False position c = (a * f(b) - a * f(a)) / (f(b) - f(a))
# def false_pos(f, a, b, tol=1e-6):
#     if (f(a) * f(b)) >= 0:
#         return "Error!"

#     while True:
#         c = (a * f(b) - b * f(a)) / (f(b) - f(a))
#         if(abs(f(c)) < tol):
#             return c

#         if(f(a) * f(c) < 0):
#             b = c
#         else:
#             a = c


# root = false_pos(f, 1, 2)

# 3. Newton raphson
# def f_p(x):
#     return 2 * x
 
# def newton_raphson(f, f_prime, n):
#     for i in range(1000):
#         if abs(f(n)) < 1e-6:
#             return n
#         n = n - (f(n) / f_prime(n))
#     raise("didn't converge")
# root = newton_raphson(f, f_p, 2)

# 4. Iteration method

# f(x) = x** 2 - 2
# def g(x):
#     return 0.5 * (x + 2/x)

# def iteration(f, g, n):
#     for i in range(1000):
#         if abs(f(n)) < 1e-6:
#             return n
#         n = g(n)              
#     raise Exception("Error!")

# root = iteration(f, g, 1)

# print(root)

x = np.linspace(-5, 5, 30)
plt.plot(x, f(x), label="f(x) = x^2 - 2")
plt.axhline(0)
plt.axvline(0)
# plt.plot(root, 0, 'ro')
plt.legend()
plt.grid()
plt.show()