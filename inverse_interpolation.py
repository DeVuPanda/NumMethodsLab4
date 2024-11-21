import numpy as np
import matplotlib.pyplot as plt

def divided_difference(x, y):

    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table

def print_divided_difference_table(x, table):

    n = len(x)
    col_width = 15

    print("\nТаблиця розділених різниць:")
    print("-" * (col_width * (n + 1)))

    header = f"{'f(x)'.center(col_width)}{'x'.center(col_width)}"
    for i in range(1, n):
        header += f"f[{i}]".center(col_width)
    print(header)
    print("-" * (col_width * (n + 1)))

    for i in range(n):
        row = f"{x[i]:.6f}".center(col_width) + f"{table[i][0]:.6f}".center(col_width)
        for j in range(1, n - i):
            row += f"{table[i][j]:.6f}".center(col_width)
        print(row)

    print("-" * (col_width * (n + 1)))


def newton_polynomial(coef, x_data, x):

    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

def get_newton_polynomial_str(x_data, y_data, coef):

    n = len(coef)
    poly_str = f"{y_data[0]:.6f}"

    for i in range(1, n):
        if y_data[i] >= 0:
            poly_str += " + "
        else:
            poly_str += " - "

        term = f"{abs(y_data[i]):.6f}"
        for j in range(i):
            term += f"(y - {x_data[j]:.6f})"

        poly_str += term

    formatted_str = ""
    line_length = 0
    for char in poly_str:
        formatted_str += char
        line_length += 1
        if line_length >= 150 and char == " ":
            formatted_str += "\n"
            line_length = 0

    return formatted_str

x = np.array([0, 0.15, 0.35, 0.6, 0.9, 1.1, 1.4, 1.57, 1.8, 2.0, 2.3, 2.6, 2.8, 3.0, 3.1])
y = np.cos(x)

x_inverse = y
y_inverse = x

div_diff_table = divided_difference(x_inverse, y_inverse)

print_divided_difference_table(x_inverse, div_diff_table)

coefficients = [div_diff_table[0][j] for j in range(len(x_inverse))]

print("\nПоліном:")
print(get_newton_polynomial_str(x_inverse, y_inverse, coefficients))

y_val = 0.5
x_found = newton_polynomial(coefficients, x_inverse, y_val)
print(f"\nЗнайдемо значення для y* = {y_val}")
print(f"Отримуємо: {x_found:.6f}")
print(f"Значення х (arccos): {np.arccos(y_val):.6f}")

x_range = np.linspace(-1, 1, 500)

y_newton = [newton_polynomial(coefficients, x_inverse, xi) for xi in x_range]

y_arccos = np.arccos(x_range)

plt.figure(figsize=(12, 7))
plt.plot(x_range, y_newton, label='Поліном отриманий методом оберненої інтерполяції', color='blue', linewidth=2)
plt.plot(x_range, y_arccos, label='arccos(x)', color='red', linestyle='--', linewidth=2)

plt.title('Порівняння f(x) = arccos(x) та поліному, отриманого методом оберненої інтерполяції', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


