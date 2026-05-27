- add button copy raw message to clipboard to menu(currently only works with markdown code blocks thanks to 3rd party library prism n co)

- voice dictation to fill prompt using local whisper models (use NPU?)

- possibility to drag and or paste image into chat(qwen3.5 is multi modal modal  potentially use subconversation/subagent to describe image and include the description into context if context management is problematic)

- max token count generation to ensure the model never go over the limit (meaning never starts truncating the prompt silently)
    => if limit is reached we should  display an error in UI

- improve specification for specifying how the UI should behave in details

