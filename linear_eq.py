# # Cramers 
import numpy as np

N = 3
arr = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
    ]

main_array = np.array(arr)
D = main_array[:, :-1]      
b = main_array[:, -1]      

# Replace each column with constants to find determinants
D1 = D.copy()
D1[:, 0] = b

D2 = D.copy()
D2[:, 1] = b

D3 = D.copy()
D3[:, 2] = b

# Calculate determinants
D_det = np.linalg.det(D)
D1_det = np.linalg.det(D1)
D2_det = np.linalg.det(D2)
D3_det = np.linalg.det(D3)

# Solve for variables
x = D1_det / D_det
y = D2_det / D_det
z = D3_det / D_det

print(x)
print(y)
print(z)

# 2. Gauss-Jordan Method


def gauss_jordan(arr, N):
    for i in range(N):
        for j in range(N):
            if i != j:
                p = arr[j][i] / arr[i][i]
                for k in range(N+1):
                    arr[j][k] -= arr[i][k] * p

    for i in range(N):
        print(arr[i][3] / arr[i][i])

gauss_jordan(arr, N)


# 3. Gauss Elimination

def gauss_elimination(arr, N):
    # Forward elimination
    for i in range(N):
        for j in range(i+1, N):
            factor = arr[j][i] / arr[i][i]
            for k in range(i, N+1):
                arr[j][k] -= arr[i][k] * factor

    # Back substitution
    x = [0] * N
    for i in range(N-1, -1, -1):
        x[i] = arr[i][N]  # start with RHS
        for j in range(i+1, N):
            x[i] -= arr[i][j] * x[j]
        x[i] /= arr[i][i]

    # Print solution
    for i in range(N):
        print(x[i])

gauss_elimination(arr, N)

# ans: [2.0 , 3.0, -1.0]