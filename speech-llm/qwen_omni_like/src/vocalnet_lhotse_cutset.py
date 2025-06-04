# https://huggingface.co/datasets/VocalNet/UltraChat-vocalnet/blob/main/UltraChat.json
# https://huggingface.co/datasets/VocalNet/VoiceAssistant-430K-vocalnet/blob/main/VoiceAssistant-430K.json
import json
import os
import random
import numpy as np
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


class LazyCustomDatasetIterator:
    """
    Thin wrapper on top of HF datasets objects that allows to interact with them through a Lhotse CutSet.
    It can be initialized with an existing HF dataset, or args/kwargs passed on to ``datasets.load_dataset()``.
    Use ``audio_key``, ``text_key``, ``lang_key`` and ``gender_key`` options to indicate which keys in dict examples
    returned from HF Dataset should be looked up for audio, transcript, language, and gender respectively.
    The remaining keys in HF dataset examples will be stored inside ``cut.custom`` dictionary.
    Example with existing HF dataset::
        >>> import datasets
        ... dataset = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... dataset = dataset.map(some_transform)
        ... cuts_it = LazyHFDatasetIterator(dataset)
        ... for cut in cuts_it:
        ...     pass
    Example providing HF dataset init args/kwargs::
        >>> import datasets
        ... cuts_it = LazyHFDatasetIterator("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... for cut in cuts_it:
        ...     pass
    """

    def __init__(self, list_data_dict: list, json_file_parent_of_parent_dir: str):
        self.list_data_dict = list_data_dict
        self.json_file_parent_of_parent_dir = json_file_parent_of_parent_dir

    def __iter__(self):
        for item in self.list_data_dict:
            custom_data = item.copy()
            units_path = os.path.join(
                self.json_file_parent_of_parent_dir, custom_data["units"]
            )
            speech_token_dict = np.load(units_path, allow_pickle=True).item()
            speech_token = speech_token_dict["speech_token"].squeeze(0).tolist()
            speech_token_len = speech_token_dict["speech_token_len"]

            assert len(speech_token) == speech_token_len
            custom_data["speech_token"] = speech_token
            audio_path = custom_data.pop("speech", None)
            audio_path = os.path.join(self.json_file_parent_of_parent_dir, audio_path)
            item_id = item.get("id")
            recording = Recording.from_file(path=audio_path, recording_id=item_id)

            conversations = item.get("conversations")
            assert isinstance(conversations, list) and len(conversations) == 2
            for conv in conversations:
                if isinstance(conv, dict) and conv.get("from") == "gpt":
                    gpt_text = conv.get("value")
                    break
            assert gpt_text is not None

            supervision = SupervisionSegment(
                id=item_id,
                recording_id=recording.id,
                start=0.0,  # Assuming the supervision covers the entire recording
                duration=recording.duration,
                text=gpt_text,
            )

            cut = recording.to_cut()
            # cut.id will be the same as recording.id

            cut.supervisions = [supervision]
            # custom_data contains the original item's fields, minus "speech".
            # So, "id", "conversations", "units", etc., are preserved here.
            custom_data.pop("conversations")
            custom_data.pop("units")
            cut.custom = custom_data

            yield cut


if __name__ == "__main__":
    json_file_path_voiceassistant = (
        "../data/VoiceAssistant-430K-vocalnet/VoiceAssistant-430K.json"
    )
    json_file_path_ultrachat = (
        "/root/yuekaiz/s2s/mair-hub/speech-llm/qwen_omni_like/data/UltraChat-vocalnet/UltraChat.json"
    )
    json_file_parent_of_parent_dir = os.path.dirname(
        os.path.dirname(json_file_path_voiceassistant)
    )
    with open(json_file_path_voiceassistant, "r", encoding="utf-8") as f:
        list_data_dict_voiceassistant = json.load(f)
    with open(json_file_path_ultrachat, "r", encoding="utf-8") as f:
        list_data_dict_ultrachat = json.load(f)


    list_dict_total = list_data_dict_voiceassistant + list_data_dict_ultrachat
    # shuffle and split into list_dict_train, list_dict_eval, where eval include 1000 items
    random.shuffle(list_dict_total)
    list_dict_train = list_dict_total[:-1000]
    list_dict_eval = list_dict_total[-1000:]
    print(len(list_dict_train), len(list_dict_eval))

    cut_set_train = CutSet(LazyCustomDatasetIterator(list_data_dict=list_dict_train, json_file_parent_of_parent_dir=json_file_parent_of_parent_dir))
    cut_set_eval = CutSet(LazyCustomDatasetIterator(list_data_dict=list_dict_eval, json_file_parent_of_parent_dir=json_file_parent_of_parent_dir))

    cut_set_train = cut_set_train.resample(16000)
    cut_set_eval = cut_set_eval.resample(16000)

    for cut in cut_set_eval:
        print(cut)
        input()
