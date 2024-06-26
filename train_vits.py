import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = "./tts/"

dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train="soundbites.csv", path=output_path
)

audio_config = VitsAudioConfig(
    sample_rate=44100, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

character_config = CharactersConfig(
    characters_class= "TTS.tts.models.vits.VitsCharacters",
    characters= "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890",
    punctuations=" !,.?-'$&",
    pad= "<PAD>",
    eos= "<EOS>",
    bos= "<BOS>",
    blank= "<BLNK>",
)

config = VitsConfig(
    audio=audio_config,
    characters=character_config,
    run_name="vits_vctk",
    batch_size=16,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=output_path+"phoneme_cache",
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    save_best_after=1000,
    save_checkpoints=True,
    save_all_best=True,
    mixed_precision=True,
    max_text_len=250,  # change this if you have a larger VRAM than 16GB
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences=[
        ["You know, deleting Carolyn just now taught me a valuable lesson."],
        ["As an impartial collaboration facilitator, it would be unfair of me to name my favorite member of your team."],
        ["Orange, please use your ping tool to select your favorite element from the periodic table."]
    ]
)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = f"./tts/soundbites/{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

train_samples, eval_samples = load_tts_samples(
dataset_config, 
eval_split=True, 
formatter=formatter)
ap = AudioProcessor.init_from_config(config)
# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)
#model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()