import sympy as sp

def fungsi(expr, x_value):
    # Evaluasi fungsi f(x) untuk nilai x tertentu.
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