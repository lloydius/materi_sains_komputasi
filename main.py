import numpy as np
import sympy as sp
import math
import pandas as pd

def matriks_nol():
    baris_nol = int(input("Masukkan jumlah baris matriks nol: "))
    kolom_nol = int(input("Masukkan jumlah kolom matriks nol: "))
    matriks = np.zeros((baris_nol, kolom_nol))
    print("\nMatriks Nol:")
    print(matriks)

def matriks_bujur_sangkar():
    ukuran_bujur_sangkar = int(input("\nMasukkan ukuran matriks bujur sangkar: "))
    matriks = np.random.randint(9, size=(ukuran_bujur_sangkar, ukuran_bujur_sangkar))
    print("\nMatriks Bujur Sangkar:")
    print(matriks)

def matriks_persegi_panjang():
    baris_persegi_panjang = int(input("\nMasukkan jumlah baris matriks persegi panjang: "))
    kolom_persegi_panjang = int(input("Masukkan jumlah kolom matriks persegi panjang: "))
    matriks = np.random.randint(9, size=(baris_persegi_panjang, kolom_persegi_panjang))
    print("\nMatriks Persegi Panjang:")
    print(matriks)

def matriks_diagonal():
    ukuran_diagonal = int(input("\nMasukkan ukuran matriks diagonal: "))
    elemen_diagonal = np.random.randint(1, 10, ukuran_diagonal)
    matriks = np.diag(elemen_diagonal)
    print("\nMatriks Diagonal:")
    print(matriks)

def matriks_identitas():
    ukuran_identitas = int(input("\nMasukkan ukuran matriks identitas: "))
    matriks = np.eye(ukuran_identitas)
    print("\nMatriks Identitas:")
    print(matriks)

def matriks_skalar():
    ukuran_skalar = int(input("\nMasukkan ukuran matriks skalar: "))
    nilai_skalar = int(input("Masukkan nilai skalar: "))
    matriks = nilai_skalar * np.eye(ukuran_skalar)
    print("\nMatriks Skalar:")
    print(matriks)

def matriks_menu():
    while True:
        print("\nMenu Jenis Matriks")
        print("1. Matriks Nol")
        print("2. Matriks Bujur Sangkar")
        print("3. Matriks Persegi Panjang")
        print("4. Matriks Diagonal")
        print("5. Matriks Identitas")
        print("6. Matriks Skalar")
        print("0. Kembali ke Menu Utama")
        pilihan_matriks = input("Pilih menu: ")

        if pilihan_matriks == "1":
            matriks_nol()
        elif pilihan_matriks == "2":
            matriks_bujur_sangkar()
        elif pilihan_matriks == "3":
            matriks_persegi_panjang()
        elif pilihan_matriks == "4":
            matriks_diagonal()
        elif pilihan_matriks == "5":
            matriks_identitas()
        elif pilihan_matriks == "6":
            matriks_skalar()
        elif pilihan_matriks == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

def interpolasi_linier(x, y, x_target):
    for i in range(len(x) - 1):
        if x[i] <= x_target <= x[i + 1]:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            return y0 + (y1 - y0) / (x1 - x0) * (x_target - x0)

def eliminasi_gauss(A, b):
    n = len(b)
    for i in range(n):
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def tabel_selisih_terbagi(x, y):
    n = len(x)
    table = [y.copy()]
    for j in range(1, n):
        column = []
        for i in range(n - j):
            diff = (table[j - 1][i + 1] - table[j - 1][i]) / (x[i + j] - x[i])
            column.append(diff)
        table.append(column)
    return table

def polinom_newton(tabel, x, nilai_x, derajat):
    hasil = tabel[0][0]
    hasil_kali = 1
    for i in range(1, derajat + 1):
        hasil_kali *= (nilai_x - x[i - 1])
        hasil += tabel[0][i] * hasil_kali
    return hasil

def interpolasi_menu():
    while True:
        print("\nMenu Interpolasi")
        print("1. Interpolasi Linier")
        print("2. Interpolasi Kuadrat")
        print("3. Interpolasi Polinom Newton")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            x = list(map(float, input("Masukkan nilai x (dipisahkan dengan spasi): ").split()))
            y = list(map(float, input("Masukkan nilai y (dipisahkan dengan spasi): ").split()))
            x_target = float(input("Masukkan nilai x target: "))
            hasil = interpolasi_linier(x, y, x_target)
            print(f"Hasil interpolasi linier untuk x = {x_target}: {hasil}")

        elif pilihan == "2":
            x_data = list(map(float, input("Masukkan nilai x (dipisahkan dengan spasi): ").split()))
            y_data = list(map(float, input("Masukkan nilai y (dipisahkan dengan spasi): ").split()))
            A = np.array([[x ** 2, x, 1] for x in x_data], dtype=float)
            b = np.array(y_data, dtype=float)
            koefisien = eliminasi_gauss(A.copy(), b.copy())
            a, b, c = koefisien
            print(f"Koefisien: a = {a}, b = {b}, c = {c}")
            x_target = float(input("Masukkan nilai x target: "))
            hasil = a * x_target**2 + b * x_target + c
            print(f"Hasil interpolasi kuadrat untuk x = {x_target}: {hasil}")

        elif pilihan == "3":
            x = list(map(float, input("Masukkan nilai x (dipisahkan dengan spasi): ").split()))
            y = list(map(float, input("Masukkan nilai y (dipisahkan dengan spasi): ").split()))
            tabel = tabel_selisih_terbagi(x, y)
            x_target = float(input("Masukkan nilai x target: "))
            derajat = int(input("Masukkan derajat polinom: "))
            hasil = polinom_newton(tabel, x, x_target, derajat)
            print(f"Hasil interpolasi polinom Newton untuk x = {x_target} dengan derajat {derajat}: {hasil}")

        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

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

def jacobi_iteration(x0, max_iterations, tolerance):
    def x1_new(x2, x3):
        return (1 / 10) * x2 - (2 / 10) * x3 + (6 / 10)

    def x2_new(x1, x3, x4):
        return (1 / 11) * x1 + (1 / 11) * x3 - (3 / 11) * x4 + (25 / 11)

    def x3_new(x1, x2, x4):
        return -(2 / 10) * x1 + (1 / 10) * x2 + (1 / 10) * x4 - (11 / 10)

    def x4_new(x2, x3):
        return -(3 / 8) * x2 + (1 / 8) * x3 + (15 / 8)

    x_old = np.array(x0, dtype=float)
    prev_norm_l2 = None
    prev_norm_inf = None

    print(f"Nilai awal: x = {x_old}")

    for k in range(1, max_iterations + 1):
        x_new = np.zeros_like(x_old)
        x_new[0] = x1_new(x_old[1], x_old[2])
        x_new[1] = x2_new(x_old[0], x_old[2], x_old[3])
        x_new[2] = x3_new(x_old[0], x_old[1], x_old[3])
        x_new[3] = x4_new(x_old[1], x_old[2])

        norm_l2 = np.sqrt(np.sum(x_new**2))
        norm_inf = np.max(np.abs(x_new))

        epsilon_l2 = abs(norm_l2 - prev_norm_l2) if prev_norm_l2 is not None else None
        epsilon_inf = abs(norm_inf - prev_norm_inf) if prev_norm_inf is not None else None

        print(f"Iterasi {k}: x = {x_new}, norm_l2 = {norm_l2:.4f}, norm_inf = {norm_inf:.4f}", end="")
        if epsilon_l2 is not None and epsilon_inf is not None:
            print(f", epsilon_l2 = {epsilon_l2:.4f}, epsilon_inf = {epsilon_inf:.4f}")
        else:
            print()

        if epsilon_l2 is not None and epsilon_inf is not None and max(epsilon_l2, epsilon_inf) < tolerance:
            print("\nSolusi konvergen:")
            print(x_new)
            print(f"Konvergensi tercapai dalam {k} iterasi.")
            return x_new

        x_old = x_new
        prev_norm_l2 = norm_l2
        prev_norm_inf = norm_inf

    print("Iterasi maksimum tercapai tanpa konvergensi.")
    return x_new

def gauss_seidel_iteration(x0, max_iterations, tolerance):
    def x1_new(x2, x3):
        return (1 / 10) * x2 - (2 / 10) * x3 + (6 / 10)

    def x2_new(x1, x3, x4):
        return (1 / 11) * x1 + (1 / 11) * x3 - (3 / 11) * x4 + (25 / 11)

    def x3_new(x1, x2, x4):
        return -(2 / 10) * x1 + (1 / 10) * x2 + (1 / 10) * x4 - (11 / 10)

    def x4_new(x2, x3):
        return -(3 / 8) * x2 + (1 / 8) * x3 + (15 / 8)

    x_old = np.array(x0, dtype=float)
    prev_norm_l2 = None
    prev_norm_inf = None

    print(f"Nilai awal: x = {x_old}")

    for k in range(1, max_iterations + 1):
        x_new = x_old.copy()
        x_new[0] = x1_new(x_new[1], x_new[2])
        x_new[1] = x2_new(x_new[0], x_new[2], x_new[3])
        x_new[2] = x3_new(x_new[0], x_new[1], x_new[3])
        x_new[3] = x4_new(x_new[1], x_new[2])

        norm_l2 = np.sqrt(np.sum(x_new**2))
        norm_inf = np.max(np.abs(x_new))

        epsilon_l2 = abs(norm_l2 - prev_norm_l2) if prev_norm_l2 is not None else None
        epsilon_inf = abs(norm_inf - prev_norm_inf) if prev_norm_inf is not None else None

        print(f"Iterasi {k}: x = {x_new}, norm_l2 = {norm_l2:.4f}, norm_inf = {norm_inf:.4f}", end="")
        if epsilon_l2 is not None and epsilon_inf is not None:
            print(f", epsilon_l2 = {epsilon_l2:.4f}, epsilon_inf = {epsilon_inf:.4f}")
        else:
            print()

        if epsilon_l2 is not None and epsilon_inf is not None and max(epsilon_l2, epsilon_inf) < tolerance:
            print("\nSolusi konvergen:")
            print(x_new)
            print(f"Konvergensi tercapai dalam {k} iterasi.")
            return x_new

        x_old = x_new
        prev_norm_l2 = norm_l2
        prev_norm_inf = norm_inf

    print("Iterasi maksimum tercapai tanpa konvergensi.")
    return x_new

def iterasi_menu():
    while True:
        print("\nMenu Iterasi")
        print("1. Iterasi Jacobi")
        print("2. Iterasi Gauss-Seidel")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih metode iterasi: ")

        if pilihan == "1" or pilihan == "2":
            x0 = list(map(float, input("Masukkan nilai awal x (dipisahkan dengan spasi): ").split()))
            max_iterations = int(input("Masukkan jumlah iterasi maksimum: "))
            tolerance = float(input("Masukkan toleransi: "))

            if pilihan == "1":
                print("\nIterasi Jacobi:")
                jacobi_iteration(x0, max_iterations, tolerance)
            else:
                print("\nIterasi Gauss-Seidel:")
                gauss_seidel_iteration(x0, max_iterations, tolerance)

        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

def distribusi_permintaan():
    n = int(input("Masukkan jumlah permintaan: "))
    permintaan = []
    probabilitas = []

    print("Masukkan nilai permintaan dan probabilitasnya (dalam format: permintaan probabilitas):")
    for _ in range(n):
        p, prob = map(float, input().split())
        permintaan.append(int(p))
        probabilitas.append(prob)

    if not np.isclose(sum(probabilitas), 1):
        print("Error: Probabilitas harus berjumlah 1.")
        return

    probabilitas_kumulatif = np.cumsum(probabilitas)

    # Tentukan interval berdasarkan probabilitas kumulatif
    interval = []
    for i in range(len(probabilitas_kumulatif)):
        awal = int(probabilitas_kumulatif[i-1] * 100) + 1 if i > 0 else 1
        akhir = int(probabilitas_kumulatif[i] * 100)
        interval.append((awal, akhir))

    # Tampilkan tabel interval bilangan acak
    print("\nTabel Interval Bilangan Acak:")
    for i, (perm, inter) in enumerate(zip(permintaan, interval)):
        print(f"Permintaan: {perm} | Interval: {inter[0]:02d} - {inter[1]:02d}")

    n_hari = int(input("\nMasukkan jumlah hari untuk simulasi: "))

    bilangan_acak = np.random.randint(1, 101, size=n_hari)

    # Tentukan permintaan berdasarkan bilangan acak
    hasil_simulasi = []
    for bilangan in bilangan_acak:
        for i, inter in enumerate(interval):
            if inter[0] <= bilangan <= inter[1]:
                hasil_simulasi.append(permintaan[i])
                break

    simulasi_df = pd.DataFrame({
        "Hari": range(1, n_hari + 1),
        "Bilangan Acak": bilangan_acak,
        "Permintaan": hasil_simulasi
    })

    print("\nTabel Simulasi:")
    print(simulasi_df)

    rata_rata = np.mean(hasil_simulasi)
    print(f"\nRata-rata permintaan per hari: {rata_rata:.2f}")

    ekspektasi = sum([p * d for p, d in zip(probabilitas, permintaan)])
    print(f"Ekspektasi (E) permintaan: {ekspektasi:.2f}")

def monte_carlo_menu():
    while True:
        print("\nMenu Monte Carlo Integration")
        print("1. Distribusi Permintaan")
        print("0. Kembali ke Menu Utama")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            distribusi_permintaan()
        elif pilihan == "0":
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

def fungsi(expr, x_value):
    "Evaluasi fungsi f(x) untuk nilai x tertentu."
    x = sp.Symbol('x')
    f = sp.sympify(expr)
    return float(f.subs(x, x_value))

def metode_tabel(expr, a, b, step):
    x = a
    print(f"{'x':^10} {'f(x)':^10}")
    print("-" * 20)
    while x <= b:
        fx = fungsi(expr, x)
        print(f"{x:^10.4f} {fx:^10.4f}")
        x += step

def metode_biseksi(expr, a, b, tolerance, max_iterations):
    x = sp.Symbol('x')
    f = sp.sympify(expr)

    for i in range(max_iterations):
        c = (a + b) / 2
        fc = fungsi(expr, c)
        print(f"Iterasi {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}, f(c)={fc:.4f}")

        if abs(fc) < tolerance or abs(b - a) < tolerance:
            return c

        if fungsi(expr, a) * fc < 0:
            b = c
        else:
            a = c
    return c

def metode_regula_falsi(expr, a, b, tolerance, max_iterations):
    for i in range(max_iterations):
        fa = fungsi(expr, a)
        fb = fungsi(expr, b)
        c = b - fb * (b - a) / (fb - fa)
        fc = fungsi(expr, c)
        print(f"Iterasi {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}, f(c)={fc:.4f}")

        if abs(fc) < tolerance:
            return c

        if fa * fc < 0:
            b = c
        else:
            a = c
    return c

def metode_iterasi_sederhana(expr, x0, tolerance, max_iterations):
    x = sp.Symbol('x')
    g = sp.sympify(expr)

    for i in range(max_iterations):
        x1 = float(g.subs(x, x0))
        print(f"Iterasi {i+1}: x0={x0:.4f}, x1={x1:.4f}")

        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1
    return x1

def metode_newton_raphson(expr, x0, tolerance, max_iterations):
    x = sp.Symbol('x')
    f = sp.sympify(expr)
    df = sp.diff(f, x)

    for i in range(max_iterations):
        fx = float(f.subs(x, x0))
        dfx = float(df.subs(x, x0))
        if dfx == 0:
            print("Turunan nol. Metode gagal.")
            return None

        x1 = x0 - fx / dfx
        print(f"Iterasi {i+1}: x0={x0:.4f}, x1={x1:.4f}, f(x0)={fx:.4f}")

        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1
    return x1

def metode_secant(expr, x0, x1, tolerance, max_iterations):
    for i in range(max_iterations):
        fx0 = fungsi(expr, x0)
        fx1 = fungsi(expr, x1)
        if fx1 - fx0 == 0:
            print("Dibagi dengan nol. Metode gagal.")
            return None

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        print(f"Iterasi {i+1}: x0={x0:.4f}, x1={x1:.4f}, x2={x2:.4f}, f(x2)={fungsi(expr, x2):.4f}")

        if abs(x2 - x1) < tolerance:
            return x2
        x0, x1 = x1, x2
    return x2

def pers_non_linier_menu():
    print("Metode Penyelesaian Persamaan Non-Linear:")
    print("1. Metode Tabel")
    print("2. Metode Biseksi")
    print("3. Metode Regula Falsi")
    print("4. Metode Iterasi Sederhana")
    print("5. Metode Newton-Raphson")
    print("6. Metode Secant")

    choice = int(input("Pilih metode (1-6): "))
    expr = input("Masukkan persamaan (contoh: x**3 - 2*x - 5): ")

    if choice == 1:
        a = float(input("Masukkan batas bawah (a): "))
        b = float(input("Masukkan batas atas (b): "))
        step = float(input("Masukkan langkah (step): "))
        metode_tabel(expr, a, b, step)

    elif choice == 2:
        a = float(input("Masukkan batas bawah (a): "))
        b = float(input("Masukkan batas atas (b): "))
        tolerance = float(input("Masukkan toleransi: "))
        max_iterations = int(input("Masukkan iterasi maksimum: "))
        result = metode_biseksi(expr, a, b, tolerance, max_iterations)
        print(f"Akar ditemukan: {result}")

    elif choice == 3:
        a = float(input("Masukkan batas bawah (a): "))
        b = float(input("Masukkan batas atas (b): "))
        tolerance = float(input("Masukkan toleransi: "))
        max_iterations = int(input("Masukkan iterasi maksimum: "))
        result = metode_regula_falsi(expr, a, b, tolerance, max_iterations)
        print(f"Akar ditemukan: {result}")

    elif choice == 4:
        x0 = float(input("Masukkan tebakan awal (x0): "))
        tolerance = float(input("Masukkan toleransi: "))
        max_iterations = int(input("Masukkan iterasi maksimum: "))
        result = metode_iterasi_sederhana(expr, x0, tolerance, max_iterations)
        print(f"Akar ditemukan: {result}")

    elif choice == 5:
        x0 = float(input("Masukkan tebakan awal (x0): "))
        tolerance = float(input("Masukkan toleransi: "))
        max_iterations = int(input("Masukkan iterasi maksimum: "))
        result = metode_newton_raphson(expr, x0, tolerance, max_iterations)
        print(f"Akar ditemukan: {result}")

    elif choice == 6:
        x0 = float(input("Masukkan tebakan awal (x0): "))
        x1 = float(input("Masukkan tebakan awal kedua (x1): "))
        tolerance = float(input("Masukkan toleransi: "))
        max_iterations = int(input("Masukkan iterasi maksimum: "))
        result = metode_secant(expr, x0, x1, tolerance, max_iterations)
        print(f"Akar ditemukan: {result}")

    else:
        print("Pilihan tidak valid.")

def main_menu():
    while True:
        print("\nMenu Utama")
        print("1. Jenis Matriks")
        print("2. Operasi Matriks")
        print("3. Interpolasi")
        print("4. Iterasi")
        print("5. Simulasi Monte Carlo")
        print("0. Keluar")
        pilihan = input("Pilih menu: ")

        if pilihan == "1":
            matriks_menu()
        elif pilihan == "2":
            matriks_operasi_menu()
        elif pilihan == "3":
            interpolasi_menu()
        elif pilihan == "4":
            iterasi_menu()
        elif pilihan == "5":
            monte_carlo_menu()
        elif pilihan == "6":
            pers_non_linier_menu()
        elif pilihan == "0":
            print("Keluar dari program.")
            break
        else:
            print("Pilihan tidak valid, silakan coba lagi.")

if __name__ == "__main__":
    main_menu()
