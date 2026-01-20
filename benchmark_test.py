from benchmark import judge_answer

def test_judge_answer():

    # Test case 1: Correct answer
    expected_output = 1  # Corresponds to "B"
    ans = "The correct option is: B"
    assert judge_answer(expected_output, ans) == 1

    # Test case 2: Incorrect answer
    expected_output = 2  # Corresponds to "C"
    ans = " option is: A (1)"
    assert judge_answer(expected_output, ans) == 0

    # Test case 3: Lowercase answer
    expected_output = 0  # Corresponds to "A"
    ans = "Answer is: a"
    assert judge_answer(expected_output, ans) == 1

    # Test case 4: No match in answer
    expected_output = 3  # Corresponds to "D"
    ans = "I am not sure about the answer."
    assert judge_answer(expected_output, ans) == 0

    # Test case 5: Answer with extra text
    expected_output = 2  # Corresponds to "C"
    ans = "answer is C: 2 "
    assert judge_answer(expected_output, ans) == 1

    # Test case 6
    expected_output = 2  # Corresponds to "C"
    ans = "answer is (C)  "
    assert judge_answer(expected_output, ans) == 1

    print("All test cases passed!")

if __name__ == "__main__":
    test_judge_answer()
