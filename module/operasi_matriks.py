import numpy as np

def gauss_jordan_elimination(matrix):
    matrix = matrix.astype(float)
    rows, cols = matrix.shape

    for i in range(rows):
        if matrix[i, i] == 0:
            raise ValueError("Elemen diagonal nol ditemukan, tidak dapat dilanjutkan.")
        matrix[i] /= matrix[i, i]

        for j in range(rows):
            if i != j:
                matrix[j] -= matrix[j, i] * matrix[i]

    return matrix

def lu_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - sum_upper

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (matrix[k][i] - sum_lower) / U[i][i]

    return L, U

def matriks_operasi_menu():
    while True:
        print("\nMenu Operasi Matriks")
        print("1. Gauss-Jordan Elimination")
        print("2. Penjumlahan Matriks")
        print("3. Perkalian Matriks")
        print("4. Determinan Matriks")
        print("5. Invers Matriks")
        print("6. Transpose Matriks")
        print("7. LU Decomposition")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            rows = int(input("Masukkan jumlah baris matriks: "))
            cols = int(input("Masukkan jumlah kolom matriks: "))

            if cols != rows + 1:
                raise ValueError("Jumlah kolom harus sama dengan jumlah baris + 1 (untuk kolom hasil).")

            print("Masukkan elemen-elemen matriks:")
            matrix = []
            for i in range(rows):
                row = list(map(float, input(f"Masukkan elemen baris {i + 1}: ").split()))
                if len(row) != cols:
                    raise ValueError(f"Jumlah elemen di baris {i + 1} harus {cols}.")
                matrix.append(row)

            matrix = np.array(matrix)
            result = gauss_jordan_elimination(matrix)
            print("\nMatriks setelah eliminasi Gauss-Jordan:")
            print(result)

            solution = result[:, -1]
            print("\nSolusi sistem persamaan:")
            print(solution)

        elif pilihan == "2":
            rows_a = int(input("Masukkan jumlah baris matriks A: "))
            cols_a = int(input("Masukkan jumlah kolom matriks A: "))
            rows_b = int(input("Masukkan jumlah baris matriks B: "))
            cols_b = int(input("Masukkan jumlah kolom matriks B: "))

            if rows_a != rows_b or cols_a != cols_b:
                print("Penjumlahan tidak dapat dilakukan. Matriks A dan B harus memiliki dimensi yang sama.")
                continue

            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i + 1} A: ").split())) for i in range(rows_a)])
            print("Masukkan elemen-elemen matriks B:")
            B = np.array([list(map(float, input(f"Masukkan baris {i + 1} B: ").split())) for i in range(rows_b)])

            C = A + B
            print("\nHasil Penjumlahan A + B:")
            print(C)

        elif pilihan == "3":
            rows_a = int(input("Masukkan jumlah baris matriks A: "))
            cols_a = int(input("Masukkan jumlah kolom matriks A: "))
            rows_b = int(input("Masukkan jumlah baris matriks B: "))
            cols_b = int(input("Masukkan jumlah kolom matriks B: "))

            if cols_a != rows_b:
                print("Perkalian tidak dapat dilakukan. Jumlah kolom A harus sama dengan jumlah baris B.")
                continue

            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i + 1} A: ").split())) for i in range(rows_a)])
            print("Masukkan elemen-elemen matriks B:")
            B = np.array([list(map(float, input(f"Masukkan baris {i + 1} B: ").split())) for i in range(rows_b)])

            D = np.dot(A, B)
            print("\nHasil Perkalian A * B:")
            print(D)

        elif pilihan == "4":
            rows_a = int(input("Masukkan jumlah baris matriks A: "))
            cols_a = int(input("Masukkan jumlah kolom matriks A: "))

            if rows_a != cols_a:
                print("Determinan hanya dapat dihitung untuk matriks persegi.")
                continue

            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i + 1} A: ").split())) for i in range(rows_a)])

            det_A = np.linalg.det(A)
            print("\nDeterminan Matriks A:", det_A)

        elif pilihan == "5":
            rows_a = int(input("Masukkan jumlah baris matriks A: "))
            cols_a = int(input("Masukkan jumlah kolom matriks A: "))

            if rows_a != cols_a:
                print("Invers hanya dapat dihitung untuk matriks persegi.")
                continue

            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i + 1} A: ").split())) for i in range(rows_a)])

            det_A = np.linalg.det(A)
            if det_A != 0:
                inv_A = np.linalg.inv(A)
                print("\nInvers Matriks A:")
                print(inv_A)
            else:
                print("\nInvers Matriks A tidak dapat dihitung karena determinan = 0.")

        elif pilihan == "6":
            rows_a = int(input("Masukkan jumlah baris matriks A: "))
            cols_a = int(input("Masukkan jumlah kolom matriks A: "))
            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i+1} A: ").split())) for i in range(rows_a)])

            transpose_A = A.T
            print("\nTranspose Matriks A:")
            print(transpose_A)

        elif pilihan == "7":
            rows = int(input("Masukkan jumlah baris matriks A: "))
            cols = int(input("Masukkan jumlah kolom matriks A: "))
            print("Masukkan elemen-elemen matriks A:")
            A = np.array([list(map(float, input(f"Masukkan baris {i+1} A: ").split())) for i in range(rows)])

            L, U = lu_decomposition(A)
            print("\nMatriks L (Lower triangular):")
            print(L)
            print("\nMatriks U (Upper triangular):")
            print(U)
            print("\nVerifikasi L * U:")
            print(np.dot(L, U))

        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")