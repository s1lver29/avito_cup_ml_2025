def main():
    print("Hello from avito-cup-ml-2025!")


if __name__ == "__main__":
    main()
from typing import Union, Tuple

class SumCalculator:
    def __init__(self, *numbers: Union[int, float]) -> None:
        self._numbers = []
        if numbers:
            for number in numbers:
                self.add_number(number)

    def calculate_sum(self) -> Union[int, float]:
        return sum(self._numbers) if self._numbers else 0

    def add_number(self, number: Union[int, float]) -> None:
        if not isinstance(number, (int, float)):
            raise TypeError(f'Expected int or float, got {type(number).__name__}')
        self._numbers.append(number)

    def clear(self) -> None:
        self._numbers.clear()

    @property
    def numbers(self) -> Tuple[Union[int, float], ...]:
        return tuple(self._numbers)

    def __str__(self) -> str:
        return f'SumCalculator(numbers={self._numbers})'

    def __repr__(self) -> str:
        return f'SumCalculator({self._numbers})'