import re

to_underscore = re.escape(" .-_,&()}{[]")
to_double_underscore = re.escape("/\\")

def slugify(text:str):
    text = re.sub(f"[{to_underscore}]", "_", text)
    text = re.sub(f"[{to_double_underscore}]", "__", text)
    return text
