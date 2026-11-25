import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

from devtools.terminal_search import interact, search


class Encoder:
    def __init__(self):
        self.questions = []

    def encode_text(self, texts):
        self.questions.extend(texts)
        if texts == ["bad"]:
            raise ValueError("TEXT_TOO_LONG")
        return np.array([[0.6, 0.8]], dtype=np.float32)


class TerminalSearchTests(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.items = [{"item_id": name, "image_name": f"{name}.png"} for name in ("a", "b", "c")]
        self.vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    def test_query_is_encoded_once_and_used_for_ranking(self):
        query, rows = search(self.encoder, self.items, self.vectors, "내가 입력한 질문", 2)
        self.assertEqual(self.encoder.questions, ["내가 입력한 질문"])
        self.assertEqual([row[0]["item_id"] for row in rows], ["b", "a"])
        for item, vector, score in rows:
            np.testing.assert_array_equal(vector, self.vectors[self.items.index(item)])
            self.assertAlmostEqual(float(query @ vector), score)

    def test_retrieved_names_go_to_llm_and_only_its_answer_is_displayed(self):
        llm = Mock()
        llm.answer.return_value = "후보로 b.png, a.png, c.png를 추천합니다."
        output = io.StringIO()
        with patch("builtins.input", side_effect=["", "bad", "버스가 있는 사진", "종료"]), redirect_stdout(output):
            result = interact(self.encoder, self.items, self.vectors, llm)
        self.assertEqual(result, 0)
        llm.answer.assert_called_once_with("버스가 있는 사진", ["b.png", "a.png", "c.png"])
        self.assertIn("답변 > " + llm.answer.return_value, output.getvalue())
        self.assertIn("TEXT_TOO_LONG", output.getvalue())
        self.assertNotIn("임베딩 전체", output.getvalue())
        self.assertNotIn("코사인 유사도=", output.getvalue())

    def test_llm_failure_is_reported_without_a_fake_generated_answer(self):
        llm = Mock()
        llm.answer.side_effect = RuntimeError("model unavailable")
        output = io.StringIO()
        with patch("builtins.input", side_effect=["버스", "/exit"]), redirect_stdout(output):
            interact(self.encoder, self.items, self.vectors, llm)
        self.assertIn("LLM 답변 생성 실패", output.getvalue())
        self.assertNotIn("답변 >", output.getvalue())
