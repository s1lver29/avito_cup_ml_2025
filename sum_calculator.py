from typing import Union, Tuple

class SumCalculator:
    def __init__(self, *numbers: Union[int, float]) -> None:
        """ Инициализация калькулятора суммы.
        Args:
            *numbers: Произвольное количество чисел (int или float) для суммирования
        Raises:
            TypeError: Если хотя бы один аргумент не является int или float
        """
        self.numbers_list = []
        if numbers:
            self.add_numbers(*numbers)

    def add_numbers(self, *numbers: Union[int, float]) -> None:
        for number in numbers:
            if not isinstance(number, (int, float)):
                raise TypeError(f"Ожидалось int или float, получено {type(number).__name__}")
            self.numbers_list.append(number)

    def calculate_sum(self) -> Union[int, float]:
        """ Вычисление суммы всех чисел.
        Returns:
            Сумма всех чисел. Возвращает 0, если чисел нет.
        """
        return sum(self.numbers_list)

    def add_number(self, number: Union[int, float]) -> None:
        """ Добавление нового числа в калькулятор.
        Args:
            number: Число для добавления (int или float)
        Raises:
            TypeError: Если number не является int или float
        """
        if not isinstance(number, (int, float)):
            raise TypeError(f"Ожидалось int или float, получено {type(number).__name__}")
        self.numbers_list.append(number)

    def clear(self) -> None:
        """ Очистка всех чисел из калькулятора. """
        self.numbers_list.clear()

    @property
    def numbers(self) -> Tuple[Union[int, float], ...]:
        """ Получение текущего списка чисел.
        Returns:
            Кортеж с текущими числами
        """
        return tuple(self.numbers_list)

    def __str__(self) -> str:
        return f'SumCalculator(numbers={self.numbers})'

    def __repr__(self) -> str:
        return f'SumCalculator(numbers={self.numbers})'