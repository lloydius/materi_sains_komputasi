import numpy as np

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