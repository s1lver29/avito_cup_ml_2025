def main():
    print("Hello from avito-cup-ml-2025!")


if __name__ == "__main__":
    main()
from abc import ABC, abstractmethod


class SumCalculator(ABC):
    @abstractmethod
    def calculate(self, a: int, b: int) -> int:
        pass