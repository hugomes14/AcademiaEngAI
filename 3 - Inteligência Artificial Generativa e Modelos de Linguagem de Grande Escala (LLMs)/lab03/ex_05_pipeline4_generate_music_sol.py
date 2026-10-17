from transformers import pipeline
import scipy

music_pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    model_kwargs={"attn_implementation": "eager"}
)

prompt = "90s rock song with electric guitar and heavy drums"
temperature = 0.8  
forward_params = {
    "temperature": temperature,
    "max_new_tokens": 70,  
}
music = music_pipe(prompt, forward_params=forward_params)
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
