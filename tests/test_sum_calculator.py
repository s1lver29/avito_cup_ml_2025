import pytest
from sum_calculator import SumCalculator


def test_empty_initialization():
    calc = SumCalculator()
    assert calc.calculate_sum() == 0


def test_single_number():
    calc = SumCalculator(5)
    assert calc.calculate_sum() == 5


def test_multiple_numbers():
    calc = SumCalculator(1, 2, 3.5)
    assert calc.calculate_sum() == 6.5


def test_invalid_type_raises_error():
    with pytest.raises(TypeError):
        SumCalculator(1, '2')


def test_sum_of_integers():
    calc = SumCalculator(1, 2, 3)
    assert calc.calculate_sum() == 6


def test_sum_of_floats():
    calc = SumCalculator(1.1, 2.2)
    assert calc.calculate_sum() == 3.3


def test_sum_mixed_types():
    calc = SumCalculator(1, 2.0, 3)
    assert calc.calculate_sum() == 6.0


def test_sum_empty():
    calc = SumCalculator()
    assert calc.calculate_sum() == 0


def test_add_valid_number():
    calc = SumCalculator()
    calc.add_number(5)
    assert calc.calculate_sum() == 5


def test_add_invalid_number():
    calc = SumCalculator()
    with pytest.raises(TypeError):
        calc.add_number('5')


def test_clear_functionality():
    calc = SumCalculator(1, 2, 3)
    calc.clear()
    assert calc.calculate_sum() == 0


def test_large_numbers():
    calc = SumCalculator(1e18, 2e18)
    assert calc.calculate_sum() == 3e18


def test_float_precision():
    calc = SumCalculator(0.1, 0.2)
    assert abs(calc.calculate_sum() - 0.3) < 1e-9


def test_immutability_of_numbers():
    calc = SumCalculator(1, 2)
    lst = calc.numbers
    with pytest.raises(AttributeError):
        lst[0] = 3