import time
import unittest
import requests
from PIL import Image
from ban_kvqa import VQA


class SomeTest(unittest.TestCase):
    def setUp(self):
        self.image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)
        self.query = "화면에 뭐가 보여?"
        self.vqa = VQA()
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_build(self):
        vqa = VQA()
        self.assertTrue(vqa)

    def test_inference_from_url(self):
        result = self.vqa.answer(self.query, self.image_url)
        print(result)
        self.assertTrue(result)

    def test_inference_from_PILIMAGE(self):
        result = self.vqa.answer(self.query, self.image)
        print(result)
        self.assertTrue(result)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SomeTest)
    unittest.TextTestRunner(verbosity=0).run(suite)