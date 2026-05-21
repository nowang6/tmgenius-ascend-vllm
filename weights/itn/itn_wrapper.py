"""
ITN模型包装器 - FST逆正则化
"""
import os


class ITNProcessor:
    """ITN (Inverse Text Normalization) 处理器"""

    def __init__(self, model_path=None, lang="zh"):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "fst_itn_zh"
            )
        self.model_path = os.path.abspath(model_path)
        self.lang = lang
        self._normalizer = self._load_normalizer()

    def _load_normalizer(self):
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"ITN model directory not found: {self.model_path}")

        tagger_path = os.path.join(self.model_path, "zh_itn_tagger.fst")
        verbalizer_path = os.path.join(self.model_path, "zh_itn_verbalizer.fst")
        if not os.path.exists(tagger_path) or not os.path.exists(verbalizer_path):
            raise FileNotFoundError(
                "Missing required ITN FST files under model directory: "
                f"{self.model_path}"
            )

        try:
            from itn.chinese.inverse_normalizer import InverseNormalizer
        except ImportError:
            raise
        except Exception as exc:
            raise ImportError(
                "WeTextProcessing failed to load. Underlying error: {}".format(exc)
            ) from exc

        if self.lang == "zh":
            return InverseNormalizer(cache_dir=self.model_path)
        return InverseNormalizer(lang=self.lang, cache_dir=self.model_path)

    def process(self, text):
        """
        对文本进行逆正则化处理

        Args:
            text: 输入文本

        Returns:
            normalized_text: 逆正则化后的文本
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if text == "":
            return ""
        return self._normalizer.normalize(text).strip()
