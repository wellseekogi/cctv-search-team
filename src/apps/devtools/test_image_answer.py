import io
import json
import unittest
from unittest.mock import Mock

from devtools.image_answer import ImageAnswerLLM


class ImageAnswerTests(unittest.TestCase):
    def setUp(self):
        self.llm = ImageAnswerLLM("llama-server.exe", "model.gguf")
        self.llm.process = Mock()
        self.llm.process.poll.return_value = None
        self.llm.opener = Mock()
        self.names = ["000000000001.jpg", "000000000002.jpg", "000000000003.jpg"]

    def respond(self, text, reason="stop"):
        body = {"choices": [{"finish_reason": reason, "message": {"content": json.dumps({"answer": text})}}]}
        self.llm.opener.open.return_value = io.BytesIO(json.dumps(body).encode())

    def test_question_and_retrieved_names_are_actual_llm_context(self):
        text = "추천 후보는 " + ", ".join(self.names) + "입니다."
        self.respond(text)
        self.assertEqual(self.llm.answer("버스 사진", self.names), text)
        request = self.llm.opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(json.loads(payload["messages"][1]["content"]),
                         {"question": "버스 사진", "image_names": self.names})

    def test_missing_or_invented_image_names_are_rejected(self):
        for text in ("사진을 추천합니다.", ", ".join(self.names + ["fake.jpg"]), ", ".join(reversed(self.names))):
            with self.subTest(text=text):
                self.respond(text)
                with self.assertRaisesRegex(ValueError, "이미지 이름"):
                    self.llm.answer("버스", self.names)

    def test_incomplete_generation_is_rejected(self):
        self.respond(", ".join(self.names), reason="length")
        with self.assertRaisesRegex(ValueError, "완성"):
            self.llm.answer("버스", self.names)

    def test_stopped_model_is_not_called(self):
        self.llm.process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            self.llm.answer("버스", self.names)
        self.llm.opener.open.assert_not_called()
