
- display error on UI when an error happens  in the websocket flow(currently nothing happens on frontend)
    also log that error in the backend console.
    
- add button copy raw message to clipboard to menu(currently only works with markdown code blocks thanks to 3rd party library prism n co)

- voice dictation to fill prompt using local whisper models (use NPU?)

- possibility to drag and or paste image into chat(use multi modal model to describe image and include the description into context, tool for the main agent to prompt the multimodal model about certain details or specific question on the image)

- max token count generation to ensure the model never go over the limit (meaning never starts truncating the prompt silently)
    => if limit is reached we should  display an error in UI

- improve specification for specifying how the UI should behave in details

- efficient way to optionally ignore venv and node_modules when using grep/glob/list_directory tools to not bloat context with unwanted stuff
