import kss
from .patterns import (
    remove_news_regex_list,
    remove_news_keyword_list,
    remove_noise_regex_list,
    multi_space_regex,
)


class Preprocessor:
    @staticmethod
    def _prep_news(x: str) -> str:
        for kwd in remove_news_keyword_list:
            x = x.replace(kwd, " ")
        for rgx in remove_news_regex_list:
            x = rgx.sub(" ", x)
        return x.strip()

    @staticmethod
    def _prep_base(x: str) -> str:
        for rgx in remove_noise_regex_list:
            x = rgx.sub(" ", x)
        return x.strip()

    @staticmethod
    def preprocess(string: str, to_lower: bool, split: bool):
        if to_lower:
            string = string.lower()
        x = Preprocessor._prep_base(string)
        x = Preprocessor._prep_news(x)
        x = multi_space_regex.sub(" ", x)
        if split:
            return kss.split_sentences(x)
        return x.strip()
