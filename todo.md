- voice dictation to fill prompt using local whisper models

- possibility to drag and or paste image into chat( use multi modal model to describe image and include the description into context, tool for the main agent to prompt the multimodal model about certain details or specific question on the image)

- better display of tools action in chat (with similar ideas to log_messages)

- max token count generation to ensure the model never go over the limit (meaning never starts truncating the prompt silently)
    => if limit is reached we should  display an error in UI


- stop closing think bubbles automatically : ALL CLOSED BY DEFAULT and if user open one it remains open 


- what is the logic behind the validate method in BaseTool and tools implementation  (why is this function called validate?????) (it must either change of be seriously explained.)


- improve specification for specifying how the UI should behave in details
