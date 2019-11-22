def text_to_training(text, context_before, context_after=None):
    if context_after is None:
        context_after = context_before
    data = ['']*context_before+list(text)+['']*(context_after+1)
    for i, c in enumerate(text):
        yield data[i:i+context_before]+data[i+context_before+1:i+context_before+context_after+1], c
    pad_front = ['']*(context_before)+list(text)
    yield pad_front[-context_before:]+['']*context_after, ''
