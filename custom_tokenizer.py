import json
from typing import Iterator, List, Union

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.processors import TemplateProcessing

tokenizer_type_dict = {
    'BPE' : [BPE(), trainers.BpeTrainer],
    'Unigram' : [Unigram(), trainers.UnigramTrainer],
    'WordLevel' : [WordLevel(), trainers.WordLevelTrainer],
    'WordPiece' : [WordPiece(), trainers.WordPieceTrainer],
}
# https://github.com/huggingface/tokenizers/tree/main/bindings/python/py_src/tokenizers/implementations
# https://github.com/huggingface/tokenizers/blob/5f6e9784526a4cd5e4f6dcdcc045cdceba5463e1/bindings/python/py_src/tokenizers/trainers/__init__.pyi

normalizers_dict = {
    'Nmt' : normalizers.Nmt(),
    'NFKC' : normalizers.NFKC(),
    'Replace' : normalizers.Replace(Regex(" {2,}"), " "),
    'Lowercase' : normalizers.Lowercase(),
}

class SentencePieceCustomTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`__ .
    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        tokenizer_type,
        normalizers_keys,
        special_token_dict,
        TemplateProcessing_dict,
        replacement: str = "▁",
        add_prefix_space: bool = True,
    ):
        self.tokenizer_type = tokenizer_type

        self.special_tokens = special_token_dict

        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]

        tokenizer = Tokenizer(tokenizer_type_dict[self.tokenizer_type][0])

        tokenizer.normalizer = normalizers.Sequence(
            [ normalizers_dict[key] for key in normalizers_keys]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)



        tokenizer.post_processor = TemplateProcessing(**TemplateProcessing_dict)

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """
        Train the model using the given files
        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training
            vocab_size (:obj:`int`):
                The size of the final vocabulary, including all tokens and alphabet.
            show_progress (:obj:`bool`):
                Whether to show progress bars while training.
            special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
                A list of special tokens the model should know of.
            initial_alphabet (:obj:`List[str]`, `optional`):
                A list of characters to include in the initial alphabet, even
                if not seen in the training dataset.
                If the strings contain more than one character, only the first one
                is kept.
            unk_token (:obj:`str`, `optional`):
                The unknown token to be used by the model.
        """

        trainer = tokenizer_type_dict[self.tokenizer_type][1](
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

        self.add_unk_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""

        trainer = tokenizer_type_dict[self.tokenizer_type][1](
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())

        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]

        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))