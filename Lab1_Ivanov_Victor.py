import math
from prettytable import PrettyTable
import mplcyberpunk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

#######################################################################################################################
code_to_test = """
import math
from prettytable import PrettyTable
import mplcyberpunk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Завдання а)")


def Horner(x):
    '''Формування таблиці
    Спочатку формуємо початкову таблицю.Після створюємо список коєфіцієнтів та робимо
    пустий список для ітераційного заповнення в циклі поліномів'''
    table = PrettyTable()
    a = [12.78, 14.35, 17.19, 1.34, -1.72]  # коефіцієнти рівняння
    table.field_names = [' ', 12.78, 14.35, 17.19, 1.34, -1.72]  # заповнення коєфіцієнтами таблиці
    ar = []
    result = a[0]
    ar.extend([x, a[0]])
    ar1 = []
    ar1.extend([x, a[0]])
    for i in range(1, len(a)):  # схема Горнера
        result = (x * result + a[i])
        ar.append(result)
        result1 = (result + a[i])
        ar1.append(result1)

    table.add_rows([ar, ar1])
    print('Таблиця Горнера:')
    print(table)
    return sum(e * x ** i for i, e in enumerate([-1.72, 1.34, 17.19, 14.35, 12.78]))


print("Залишок:", Horner(3.25))

plt.style.use('cyberpunk')
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style('darkgrid')

bbox_properties = dict(
    boxstyle="Round, pad=0.4",
    ec="k",
    fc="w",
    ls="-",
    lw=1)

x = 3.25
y1 = 12.78 * x ** 4 + 14.35 * x ** 3 + 17.19 * x * x + 1.34 * x - 1.72
x = np.linspace(-4, 4, 50)
y = 12.78 * x ** 4 + 14.35 * x ** 3 + 17.19 * x * x + 1.34 * x - 1.72

ln = sns.lineplot(x=x, y=y, linewidth=4, color='red')  # формування кольорів ліній
ln.set_xlabel('X', fontsize=12)  # підписи осей
ln.set_ylabel('Y', fontsize=12)
plt.axvline(x=3.25)
f = '''r'$y=12.78x^4 + 14.35x^3 + 17.19x^2+1.34x - 1.72$' + '\n' + f'y(3.25) = {y1}\n'''
plt.text(-4, 4000, f, fontsize=12, bbox=bbox_properties)  # рівняння

mplcyberpunk.add_glow_effects()
plt.show()
fig.savefig('saved_figure.png', dpi=300, bbox_inches='tight')

print("Завдання б)")


def my_sin(x, eps=1.0e-14):
    '''Функція синуса
    an = (((-1)**n)*x**(2*n+1))/(2*n+1)! при n = 0,1,...,inf
    an+1 = (((-1)**(n+1))*x**(2*n+3))/(2*n+3)!
    Тепер отримаємо рекурентну формулу, що зв'язує an+1 з an:
    an+1/an = ((-1)*x**2)/((2*n+2)(2*n+3))
    Тоді an+1 = (-an*x**2)/((2*n+2)(2*n+3))
    Ця формула працює в коді.Тільки n в коді змінюється через 2'''
    n = 1
    a = x
    s = 0
    while abs(a) > eps:
        s += a
        a = -a * x * x / ((n + 1) * (n + 2))
        n += 2
    return s


mytable = PrettyTable()
mytable.field_names = ["x", "my_sin", "sin", "diff"]

for k in range(16):
    x = 0.345 + 0.005 * k
    a = my_sin(x)
    b = math.sin(x)
    # print(x,a , b, abs(a-b))
    mytable.add_row([x, a, b, abs(a - b)])

print("Таблиця значень функції sin(x)")
print(mytable)

print("Завдання в)")
print("Таблиця значень функції Y=1/(x**3)")

vtablev = PrettyTable()
vtablev.field_names = ["k", "y"]

for k in range(0, 16):
    y = 1/((3+2*k)**3)
    y = round(y, 15)
    # print('k = {}; y = {}'.format(k, y))
    vtablev.add_row([k, y])

print(vtablev)
"""

elapsed_time = timeit.timeit(code_to_test, number=0)
print("Час виконання кода:", elapsed_time, 'ms')
#######################################################################################################################

'''====================================================================================================================
                                            Лабороторна робота №1.Обчислення значень функції
Мета: закріплення знань із застосування прийомів, що зводять обчислення 
деяких функцій до послідовності елементарних операцій

Роботу виконував: Іванов Віктор Віталійович, ФФ-03
Роботу перевіряла: Гордійко Н.О.
Варіант №15
Завдання:
а) Обчислити значення многочлена:
P(x) = 12.78(x**4) + 14.35(x**3) + 17.19(x**2) + 1.34x - 1.72 при x = 3.25;
б) Скласти таблицю значень функції sin x з точністю до 10**(-15),
x = 0.345 + 0.005k (k = 0,1,2,...,15),користуючись її розкладом в степеневий ряд;
в) Скласти таблицю значень функції у з точністю до 10**(-15), користуючись методом ітерацій, якщо:
y = 1/(x**3), x = 3 + 2k (k = 0,1,2,...,15).
===================================================================================================================='''

'---------------------------------------------------------------------------------------------------------------------'
"Програмний код"

"Документація кода"

print("Завдання а)")


def Horner(x):
    '''Формування таблиці
    Спочатку формуємо початкову таблицю.Після створюємо список коєфіцієнтів та робимо
    пустий список для ітераційного заповнення в циклі поліномів'''
    table = PrettyTable()
    a = [12.78, 14.35, 17.19, 1.34, -1.72]  # коефіцієнти рівняння
    table.field_names = [' ', 12.78, 14.35, 17.19, 1.34, -1.72]  # заповнення коєфіцієнтами таблиці
    ar = []
    result = a[0]
    ar.extend([x, a[0]])
    ar1 = []
    ar1.extend([x, a[0]])
    for i in range(1, len(a)):  # схема Горнера
        result = (x * result + a[i])
        ar.append(result)
        result1 = (result + a[i])
        ar1.append(result1)

    table.add_rows([ar, ar1])
    print('Таблиця Горнера:')
    print(table)
    return sum(e * x ** i for i, e in enumerate([-1.72, 1.34, 17.19, 14.35, 12.78]))


print("Залишок:", Horner(3.25))

plt.style.use('cyberpunk')
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style('darkgrid')

bbox_properties = dict(
    boxstyle="Round, pad=0.4",
    ec="k",
    fc="w",
    ls="-",
    lw=1)

x = 3.25
y1 = 12.78 * x ** 4 + 14.35 * x ** 3 + 17.19 * x * x + 1.34 * x - 1.72
x = np.linspace(-4, 4, 50)
y = 12.78 * x ** 4 + 14.35 * x ** 3 + 17.19 * x * x + 1.34 * x - 1.72

ln = sns.lineplot(x=x, y=y, linewidth=4, color='red')  # формування кольорів ліній
ln.set_xlabel('X', fontsize=12)  # підписи осей
ln.set_ylabel('Y', fontsize=12)
plt.axvline(x=3.25)
f = r'$y=12.78x^4 + 14.35x^3 + 17.19x^2+1.34x - 1.72$' + '\n' + f'y(3.25) = {y1}\n'
plt.text(-4, 4000, f, fontsize=12, bbox=bbox_properties)  # рівняння

mplcyberpunk.add_glow_effects()
plt.show()
fig.savefig('saved_figure.png', dpi=300, bbox_inches='tight')

print("Завдання б)")


def my_sin(x, eps=1.0e-14):
    '''Функція синуса
    Ряд Маклорена
    an = (((-1)**n)*x**(2*n+1))/(2*n+1)! при n = 0,1,...,inf
    an+1 = (((-1)**(n+1))*x**(2*n+3))/(2*n+3)!
    Тепер отримаємо рекурентну формулу, що зв'язує an+1 з an:
    an+1/an = ((-1)*x**2)/((2*n+2)(2*n+3))
    Тоді an+1 = (-an*x**2)/((2*n+2)(2*n+3))
    Ця формула працює в коді.Тільки n в коді змінюється через 2'''
    n = 1
    a = x
    s = 0
    while abs(a) > eps:
        s += a
        a = -a * x * x / ((n + 1) * (n + 2))
        n += 2
    return s


mytable = PrettyTable()
mytable.field_names = ["x", "my_sin", "sin", "diff"]

for k in range(16):
    x = 0.345 + 0.005 * k
    a = my_sin(x)
    b = math.sin(x)
    # print(x,a , b, abs(a-b))
    mytable.add_row([x, a, b, abs(a - b)])

print("Таблиця значень функції sin(x)")
print(mytable)

print("Завдання в)")
print("Таблиця значень функції Y=1/(x**3)")

vtablev = PrettyTable()
vtablev.field_names = ["k", "y"]

for k in range(0, 16):
    y = 1/((3+2*k)**3)
    y = round(y, 15)
    # print('k = {}; y = {}'.format(k, y))
    vtablev.add_row([k, y])

print(vtablev)
'---------------------------------------------------------------------------------------------------------------------'

"""
Висновок: було реалізовано різні алгоритми кода для застосування прийомів, що зводять обчислення 
деяких функцій до послідовності елементарних операцій, також було виміряно швидкість коду, а також у котрий раз
ми впевнелися, що найкращой перевіркою достовірності методу - числова відповідь - графічний розв'язок рівняння.
"""
