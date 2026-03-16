import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# 1. The "Source of Truth" from your database/PDFs
KNOWLEDGE_BASE = [
    "Clients can dispute credit card transactions within 60 days of the statement date.",
    "Disputes can be filed online through the RBC Mobile app or by calling customer service."
]

def test_chatbot_hallucination():
    # 2. Simulate what the Chatbot actually said (The "Actual Output")
    # Let's pretend the chatbot hallucinated and said "90 days" instead of "60 days"
    user_input = "How long do I have to dispute a charge?"
    actual_output = "You have 90 days to dispute a transaction via the RBC app."
    
    # 3. Define the Metrics
    # Faithfulness checks if the output is supported by the KNOWLEDGE_BASE
    faithfulness_metric = FaithfulnessMetric(threshold=0.7)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

    # 4. Create the Test Case
    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=KNOWLEDGE_BASE
    )

    # 5. Execute the test
    # This will FAIL because 90 != 60, and DeepEval will explain WHY.
    assert_test(test_case, [faithfulness_metric, relevancy_metric])
