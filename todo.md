- add button copy raw message to clipboard to menu(currently only works with markdown code blocks thanks to 3rd party library prism n co)

- GPU/CPU memory view in UI: add a panel or status indicator showing VRAM used/free and how many layers are on GPU vs CPU, by calling nvidia-smi from the backend and exposing it via an endpoint

- voice dictation to fill prompt using local whisper models (use NPU?)

- possibility to drag and or paste image into chat(qwen3.5 is multi modal modal  potentially use subconversation/subagent to describe image and include the description into context if context management is problematic)

- max token count generation to ensure the model never go over the limit (meaning never starts truncating the prompt silently)
    => if limit is reached we should  display an error in UI

- improve specification for specifying how the UI should behave in details



- continue investigating with tokenization difference with ollama
(file tested in diagnose/ folder) You concluded that there was a fundamental difference in BPE tokenizer between the one we build and the one in llama.cpp (i dont know if that conclusion was absolutely true, it might have been biased). You were about to try to install llama-cpp-python to see what could be done directly through that api( though i dont remember the details)

also choosing to swith to llama.cpp as a backend could be a good idea as it could (maybe? idk) help us actually stream the content of large tool calls like write file / edit_file with large diffs to better understand what it is doing in intterrupt if appropriate


- in the frontend check that the call to count token on messages is called before the agent loop finishes