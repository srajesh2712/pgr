"""A simple interactive maths calculator.

The calculator accepts two numbers, including decimal values, and performs
one of four basic operations selected by the user.
"""


def calculate(first_number: float, operation: str, second_number: float) -> float:
    """Return the result of applying an operation to two numbers."""
    if operation == "1":
        return first_number + second_number
    if operation == "2":
        return first_number - second_number
    if operation == "3":
        return first_number * second_number
    if operation == "4":
        # Division by zero is invalid, so report it clearly to the caller.
        if second_number == 0:
            raise ValueError("Cannot divide by zero.")
        return first_number / second_number
    raise ValueError("Please choose an operation from 1 to 4.")


def main() -> None:
    """Read user input and display the calculation result."""
    print("Simple Maths Calculator")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    try:
        # float() allows both whole numbers and decimal values.
        operation = input("Choose an option (1-4): ").strip()
        first_number = float(input("Enter the first number: "))
        second_number = float(input("Enter the second number: "))

        result = calculate(first_number, operation, second_number)
        print(f"Result: {result}")
    except ValueError as error:
        # Display input and calculation errors without showing a traceback.
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
