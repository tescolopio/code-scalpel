"""
Test Generation Demo: Loan Approval Algorithm

This demo proves Code Scalpel's symbolic execution engine can analyze
branching logic and generate concrete test inputs for EVERY path.

Run:
    code-scalpel analyze demos/test_gen_scenario.py
    
Then use the MCP server:
    generate_unit_tests(code, function_name="loan_approval")

Expected Output: pytest cases with exact values like:
    - test_reject: credit_score=599
    - test_instant_approve: income=100001, debt=4999, credit_score=700
    - test_manual_review: income=100001, debt=5001, credit_score=700
    - test_high_risk: income=50000, debt=30000, credit_score=700
    - test_standard_approve: income=50000, debt=20000, credit_score=700
"""


def loan_approval(income: int, debt: int, credit_score: int) -> str:
    """
    Determine loan approval status based on financial metrics.

    This function has 5 distinct paths that symbolic execution should find:

    Path 1: REJECT (credit_score < 600)
    Path 2: INSTANT_APPROVE (income > 100000 AND debt < 5000)
    Path 3: MANUAL_REVIEW (income > 100000 AND debt >= 5000)
    Path 4: HIGH_RISK (debt > income * 0.5)
    Path 5: STANDARD_APPROVE (default case)

    Args:
        income: Annual income in dollars
        debt: Total debt in dollars
        credit_score: FICO score (300-850)

    Returns:
        Approval status string
    """
    # Path 1: Immediate rejection for low credit
    if credit_score < 600:
        return "REJECT"

    # Path 2 & 3: High income branch
    if income > 100000:
        if debt < 5000:
            return "INSTANT_APPROVE"  # Path 2: Rich and debt-free
        else:
            return "MANUAL_REVIEW"  # Path 3: Rich but has debt

    # Path 4: Debt-to-income ratio check
    if debt > (income * 0.5):
        return "HIGH_RISK"

    # Path 5: Default approval
    return "STANDARD_APPROVE"


def calculate_interest_rate(credit_score: int, loan_amount: int) -> float:
    """
    Calculate interest rate based on credit score and loan amount.

    Another function for symbolic execution to analyze.
    Has 4 paths based on credit score tiers.
    """
    # Base rate
    base_rate = 5.0

    # Credit score adjustments
    if credit_score >= 800:
        rate_adjustment = -1.5  # Excellent credit discount
    elif credit_score >= 700:
        rate_adjustment = -0.5  # Good credit discount
    elif credit_score >= 600:
        rate_adjustment = 1.0  # Fair credit premium
    else:
        rate_adjustment = 3.0  # Poor credit premium

    # Large loan adjustment
    if loan_amount > 500000:
        rate_adjustment += 0.25

    return base_rate + rate_adjustment


# Self-test to verify the logic
if __name__ == "__main__":
    # These are the exact values Z3 should derive
    test_cases = [
        (50000, 10000, 599, "REJECT"),  # Path 1
        (100001, 4999, 700, "INSTANT_APPROVE"),  # Path 2
        (100001, 5001, 700, "MANUAL_REVIEW"),  # Path 3
        (50000, 30000, 700, "HIGH_RISK"),  # Path 4
        (50000, 20000, 700, "STANDARD_APPROVE"),  # Path 5
    ]

    print("Loan Approval Logic Verification:")
    print("-" * 50)
    for income, debt, score, expected in test_cases:
        result = loan_approval(income, debt, score)
        status = "✓" if result == expected else "✗"
        print(f"  {status} income={income}, debt={debt}, score={score}")
        print(f"      Expected: {expected}, Got: {result}")
    print("-" * 50)
    print(
        "If Code Scalpel's test generator works, it will derive these values automatically."
    )
